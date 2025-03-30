# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


from diffusers.utils import is_accelerate_available

from ..models.unet import UNetModel
from ..models.autoencoder import AutoencoderKL, AutoencoderKL_Dualref
from ..models.condition import FrozenOpenCLIPEmbedder, FrozenOpenCLIPImageEmbedderV2, Resampler
from ..models.layer_controlnet import LayerControlNet

from diffusers.schedulers import DDIMScheduler
from diffusers.utils import BaseOutput, logging
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg

from einops import rearrange


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[List[Image.Image], np.ndarray]


class AnimationPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]
    def __init__(
        self,
        vae,
        vae_dualref,
        text_encoder,
        image_encoder,
        image_projector,
        unet: UNetModel,
        layer_controlnet: LayerControlNet,
        scheduler: DDIMScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            vae_dualref=vae_dualref,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            image_projector=image_projector,
            unet=unet,
            layer_controlnet=layer_controlnet,
            scheduler=scheduler,
        )
        if vae is not None:
            self.vae_scale_factor = 2 ** (len(self.vae.config.ddconfig["ch_mult"]) - 1)
        else:
            self.vae_scale_factor = 2 ** (len(self.vae_dualref.config.ddconfig["ch_mult"]) - 1)

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.layer_encoder, self.text_encoder, self.vae, self.vae_dualref]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_embeddings = self.text_encoder(prompt)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_embeddings = self.text_encoder(uncond_tokens)

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def _encode_image(self, image, device, num_videos_per_prompt, do_classifier_free_guidance):
        batch_size = image.shape[0]

        image_embeddings = self.image_encoder(image)
        image_embeddings = self.image_projector(image_embeddings)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_embeddings = self.image_encoder(torch.zeros_like(image))
            uncond_embeddings = self.image_projector(uncond_embeddings)
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and image embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([uncond_embeddings, image_embeddings])

        return image_embeddings

    def _encode_controls(
        self,
        layer_masks,
        layer_regions,
        layer_validity,
        motion_scores,
        layer_static,
        trajectories,
        sketches,
        video_length,
        mode,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance
    ):
        vae = self.vae if self.vae is not None else self.vae_dualref

        batch_size, n_layers = layer_masks.shape[:2]
        # Frame decomposition
        layer_regions = rearrange(layer_regions, "b n f c h w -> (b n f) c h w")
        keyframe_layer_latents = vae.encode(layer_regions)[0].sample() * 0.18215
        keyframe_layer_latents = rearrange(keyframe_layer_latents, "(b n f) c h w -> b n f c h w", b=batch_size, n=n_layers)
        layer_latents_shape = list(keyframe_layer_latents.shape)
        layer_latents_shape[2] = video_length
        layer_latents = torch.zeros(layer_latents_shape, device=device, dtype=keyframe_layer_latents.dtype)
        resized_layer_masks = rearrange(layer_masks, "b n f c h w -> (b n f) c h w")
        resized_layer_masks = F.interpolate(resized_layer_masks.float(), size=layer_latents.shape[-2:], mode="bilinear")
        resized_layer_masks = rearrange(resized_layer_masks, "(b n f) c h w -> b n f c h w", b=batch_size, n=n_layers).to(dtype=layer_latents.dtype)
        layer_latent_mask_shape = list(resized_layer_masks.shape)
        layer_latent_mask_shape[2] = video_length
        layer_latent_mask = torch.zeros(layer_latent_mask_shape, device=device, dtype=resized_layer_masks.dtype)

        for batch_idx in range(batch_size):
            if mode != "interpolate":
                layer_latents[batch_idx, :, 0] = keyframe_layer_latents[batch_idx, :, 0]
                layer_latent_mask[batch_idx, :, 0] = resized_layer_masks[batch_idx, :, 0]
                if layer_static[batch_idx].any():
                    static_indices = torch.nonzero(layer_static[batch_idx]).squeeze(1)
                    layer_latents[batch_idx, static_indices, :] = keyframe_layer_latents[batch_idx, static_indices, 0:1].repeat(1, video_length, 1, 1, 1)
                    layer_latent_mask[batch_idx, static_indices, :] = resized_layer_masks[batch_idx, static_indices, 0:1].repeat(1, video_length, 1, 1, 1)
            else:
                layer_latents[batch_idx, :, 0] = keyframe_layer_latents[batch_idx, :, 0]
                layer_latents[batch_idx, :, -1] = keyframe_layer_latents[batch_idx, :, -1]
                layer_latent_mask[batch_idx, :, 0] = resized_layer_masks[batch_idx, :, 0]
                layer_latent_mask[batch_idx, :, -1] = resized_layer_masks[batch_idx, :, -1]
                if layer_static[batch_idx].any():
                    static_indices = torch.nonzero(layer_static[batch_idx]).squeeze(1)
                    layer_latents[batch_idx, static_indices, :video_length//2] = keyframe_layer_latents[batch_idx, static_indices, 0:1].repeat(1, video_length//2, 1, 1, 1)
                    layer_latents[batch_idx, static_indices, video_length//2:] = keyframe_layer_latents[batch_idx, static_indices, -1:].repeat(1, video_length//2, 1, 1, 1)
                    layer_latent_mask[batch_idx, static_indices, :video_length//2] = resized_layer_masks[batch_idx, static_indices, 0:1].repeat(1, video_length//2, 1, 1, 1)
                    layer_latent_mask[batch_idx, static_indices, video_length//2:] = resized_layer_masks[batch_idx, static_indices, -1:].repeat(1, video_length//2, 1, 1, 1)
        layer_latents = torch.repeat_interleave(layer_latents, num_videos_per_prompt, dim=0)
        layer_latent_mask = torch.repeat_interleave(layer_latent_mask, num_videos_per_prompt, dim=0)
        layer_validity = torch.repeat_interleave(layer_validity, num_videos_per_prompt, dim=0)

        sketches = rearrange(sketches, 'b n f c h w -> (b n f) c h w')
        layer_sketch_latents = vae.encode(sketches)[0].sample() * 0.18215
        layer_sketch_latents = rearrange(layer_sketch_latents, '(b n f) c h w -> b n f c h w', b=batch_size, n=n_layers)
        layer_sketch_latents = torch.repeat_interleave(layer_sketch_latents, num_videos_per_prompt, dim=0)

        trajectories = torch.repeat_interleave(trajectories, num_videos_per_prompt, dim=0)

        motion_scores = torch.repeat_interleave(motion_scores, num_videos_per_prompt, dim=0)

        if do_classifier_free_guidance:
            layer_latents = torch.cat([layer_latents, layer_latents], dim=0)
            layer_latent_mask = torch.cat([layer_latent_mask, layer_latent_mask], dim=0)
            motion_scores = torch.cat([motion_scores, motion_scores], dim=0)
            layer_sketch_latents = torch.cat([layer_sketch_latents, layer_sketch_latents], dim=0)
            trajectories = torch.cat([trajectories, trajectories], dim=0)
            layer_validity = torch.cat([layer_validity, layer_validity], dim=0)
        return dict(
            layer_latents=layer_latents,
            layer_latent_mask=layer_latent_mask,
            motion_scores=motion_scores,
            sketch=layer_sketch_latents,
            trajectory=trajectories,
            layer_validity=layer_validity,
        )

    def get_latent_z_with_hidden_states(self, videos):
        b, f, c, h, w = videos.shape
        x = rearrange(videos, 'b f c h w -> (b f) c h w')
        encoder_posterior, hidden_states = self.vae_dualref.encode(x, return_hidden_states=True)
        hidden_states_first_last = []
        ### use only the first and last hidden states
        for hid in hidden_states:
            hid = rearrange(hid, '(b f) c h w -> b c f h w', f=f)
            hid_new = torch.cat([hid[:, :, 0:1], hid[:, :, -1:]], dim=2)
            hidden_states_first_last.append(hid_new.float())

        z = encoder_posterior[0].sample() * 0.18215
        z = rearrange(z, '(b f) c h w -> b c f h w', b=b, f=f).detach()
        return z, hidden_states_first_last

    def get_latent_z(self, videos):
        b, f, c, h, w = videos.shape
        x = rearrange(videos, 'b f c h w -> (b f) c h w')
        z = self.vae.encode(x)[0].sample() * 0.18215
        z = rearrange(z, '(b f) c h w -> b c f h w', b=b, f=f).detach()
        return z

    def decode_latents(self, latents):
        batch_size = latents.shape[0]
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for batch_idx in range(batch_size):
            video.append(self.vae.decode(latents[batch_idx * video_length:(batch_idx + 1) * video_length]).sample)
        video = torch.cat(video, dim=0)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def decode_latents_with_hidden_states(self, latents, hidden_states):
        batch_size = latents.shape[0]
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for batch_idx in range(batch_size):
            video.append(self.vae_dualref.decode(latents[batch_idx * video_length:(batch_idx + 1) * video_length].float(), ref_context=hidden_states, timesteps=video_length).sample)
        video = torch.cat(video, dim=0)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = device

            if isinstance(generator, list):
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: int,
        height: int,
        width: int,
        frame_tensor: torch.FloatTensor,
        layer_masks: torch.FloatTensor,     # [b, n_layers, 1 (2), c, h, w]
        layer_regions: torch.FloatTensor,   # [b, n_layers, 1 (2), c, h, w]
        layer_static: torch.Tensor,         # [b, n_layers]
        motion_scores: torch.Tensor,        # [b, n_layers]
        sketch: torch.FloatTensor,          # [b, n_layers, f, c, h, w]
        trajectory: torch.FloatTensor,      # [b, n_layers, f, c, h, w]
        layer_validity: torch.Tensor,       # [b, n_layers]
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        guidance_rescale: float=0.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,

        fps: Optional[int] = 24,
        mode: str = "interpolate",
        weight_dtype: torch.dtype = torch.float32,

        **kwargs,
    ):
        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = len(frame_tensor)
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        fps = torch.tensor([fps] * batch_size * num_videos_per_prompt, device=device, dtype=weight_dtype)
        frame_tensor = frame_tensor.to(dtype=weight_dtype)
        layer_regions = layer_regions.to(dtype=weight_dtype)
        motion_scores = motion_scores.to(dtype=weight_dtype)
        sketch = sketch.to(dtype=weight_dtype)
        trajectory = trajectory.to(dtype=weight_dtype)

        # Encode layer-level controls
        encoded_layer_controls = self._encode_controls(
            layer_masks,
            layer_regions,
            layer_validity,
            motion_scores,
            layer_static,
            trajectory,
            sketch,
            video_length,
            mode,
            device,
            num_videos_per_prompt,
            do_classifier_free_guidance
        )
        layer_validity = encoded_layer_controls.pop("layer_validity")

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        cond_frame = frame_tensor[:, 0] # [b, f, c, h, w] -> [b, c, h, w]
        image_embeddings = self._encode_image(
            cond_frame, device, num_videos_per_prompt, do_classifier_free_guidance
        )

        if mode == "interpolate":
            z, hidden_states = self.get_latent_z_with_hidden_states(frame_tensor)
        else:
            z = self.get_latent_z(frame_tensor)
        z = z.to(dtype=weight_dtype)
        if mode != "interpolate":
            img_cat_cond = z[:, :, :1]
            img_cat_cond = img_cat_cond.repeat(1, 1, video_length, 1, 1)
        else:
            img_cat_cond = torch.zeros_like(z[:, :, :1].repeat(1, 1, video_length, 1, 1))
            img_cat_cond[:, :, 0] = z[:, :, 0]
            img_cat_cond[:, :, -1] = z[:, :, -1]
        img_cat_cond = torch.repeat_interleave(img_cat_cond, num_videos_per_prompt, dim=0)
        if do_classifier_free_guidance:
            img_cat_cond = torch.cat([img_cat_cond, img_cat_cond], dim=0)
            fps = torch.cat([fps, fps], dim=0)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.unet.out_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_with_img = torch.cat([latent_model_input, img_cat_cond], dim=1)

                if do_classifier_free_guidance:
                    ts = torch.full((batch_size * num_videos_per_prompt * 2,), t, device=device, dtype=torch.long)
                else:
                    ts = torch.full((batch_size * num_videos_per_prompt,), t, device=device, dtype=torch.long)

                layer_features = self.layer_controlnet(
                    noise_with_img, ts,
                    context_text=text_embeddings,
                    context_img=image_embeddings,
                    fps=fps,
                    **encoded_layer_controls
                )
                noise_pred = self.unet(
                    noise_with_img, ts,
                    context_text=text_embeddings,
                    context_img=image_embeddings,
                    fps=fps,
                    controls=layer_features,
                    layer_validity=layer_validity,
                ).sample.to(dtype=weight_dtype)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_cond, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Post-processing
        if mode == "interpolate":
            video = self.decode_latents_with_hidden_states(latents, hidden_states)
        else:
            video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)
