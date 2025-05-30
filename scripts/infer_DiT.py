import argparse
import sys
import math
import json
import torch
import decord
import os
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from einops import rearrange
from tqdm.auto import tqdm

from torch.nn.functional import interpolate
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from scipy.interpolate import PchipInterpolator

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from DiT.vae import WanVAE
from DiT.model import VaceWanModel
from DiT.utils import save_videos_grid, save_videos_with_traj, generate_gaussian_heatmap
from wan.text2video import T5EncoderModel, FlowUniPCMultistepScheduler


HEIGHT = 480
WIDTH = 832
LENGTH = 81
LAYER_CAPACITY = 4
DEVICE = "cuda"
WEIGHT_DTYPE = torch.bfloat16
VAE_STRIDE = [4, 8, 8]
PATCH_SIZE = [1, 2, 2]
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize(min(HEIGHT, WIDTH)),
    transforms.CenterCrop((HEIGHT, WIDTH)),
])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON configuration file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save the output video."
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="checkpoints/LayerAnimate-DiT",
        help="Path to the pretrained model directory."
    )
    args = parser.parse_args()

    text_encoder = T5EncoderModel(
        text_len=WIDTH,
        dtype=WEIGHT_DTYPE,
        device=DEVICE,
        checkpoint_path=os.path.join(args.pretrained_model, "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=os.path.join(args.pretrained_model, "google/umt5-xxl"))
    vae = WanVAE(vae_pth=os.path.join(args.pretrained_model, "Wan2.1_VAE.pth"), dtype=WEIGHT_DTYPE, device=DEVICE)
    video_model = VaceWanModel.from_pretrained(args.pretrained_model, subfolder="transformer").to(device=DEVICE, dtype=WEIGHT_DTYPE)
    video_model.eval().requires_grad_(False)


    config  = OmegaConf.load(args.config)

    prompt = config.get("prompt", "an anime scene.")
    n_prompt = config.get("n_prompt", "")
    seed = config.get("seed", 42)
    num_inference_steps = config.get("num_inference_steps", 25)
    guidance_scale = config.get("guidance_scale", 6.0)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 加载初始帧和结束帧
    try:
        input_image = Image.open(config["first_frame_path"]).convert("RGB")
        input_image = IMG_TRANSFORM(input_image)

        input_image_end = None
        if config.get("last_frame_path"):
            input_image_end = Image.open(config["last_frame_path"]).convert("RGB")
            input_image_end = IMG_TRANSFORM(input_image_end)
    except FileNotFoundError as e:
        print(f"Error: Image file not found - {e}")
        return

    # Prepare layer parameters
    layer_masks = []
    layer_scores = []
    layer_sketches = []
    layer_trajectories = []
    assert len(config["layer"]) <= LAYER_CAPACITY, f"The number of layers in the config exceeds the maximum capacity of {LAYER_CAPACITY}."

    print("Parsing layer configurations...")
    for i in range(LAYER_CAPACITY):
        if i >= len(config["layer"]):
            layer_masks.append(None)
            layer_scores.append(-1)
            layer_sketches.append(None)
            layer_trajectories.append([[]])
            continue

        layer_config = config["layer"][i]
        mask_path = layer_config["mask_path"]
        layer_masks.append(IMG_TRANSFORM(Image.open(mask_path).convert("L")) if mask_path else None)
        control_type = layer_config.get("control_type", None)
        if control_type == "sketch":
            score = -1
            sketch = layer_config["sketch_path"]
            trajectory = [[]]
        elif control_type == "trajectory":
            score = -1
            sketch = None
            traj_path = layer_config["trajectory_path"]
            with open(traj_path, 'r') as f:
                trajectory = json.load(f)
        elif control_type == "score":
            score = layer_config["score"]
            sketch = None
            trajectory = [[]]
        else:
            raise ValueError(f"Unsupported control type: {control_type}")
        layer_scores.append(score)
        layer_sketches.append(sketch)
        layer_trajectories.append(trajectory)

    print("Starting inference...")
    run(
        text_encoder,
        vae,
        video_model,
        input_image=input_image,
        input_image_end=input_image_end,
        seed=seed,
        prompt=prompt,
        n_prompt=n_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        savedir=args.output_dir,
        input_layer_masks=layer_masks,
        input_layer_scores=layer_scores,
        input_layer_sketches=layer_sketches,
        input_layer_trajectories=layer_trajectories,
    )


@torch.no_grad()
def run(text_encoder, vae, video_model,
        input_image, input_image_end, seed, prompt, n_prompt,
        num_inference_steps, guidance_scale, savedir,
        input_layer_masks, input_layer_scores, input_layer_sketches, input_layer_trajectories):
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator(DEVICE).manual_seed(seed)
    do_classifier_free_guidance = guidance_scale > 1.0

    masked_videos = torch.zeros((1, LENGTH, 3, HEIGHT, WIDTH), dtype=WEIGHT_DTYPE, device=DEVICE)
    context_frame_masks = torch.zeros_like(masked_videos)
    image1 = F.to_tensor(input_image) * 2 - 1
    masked_videos[0, 0] = image1.to(DEVICE)
    context_frame_masks[0, 0] = 1.0
    if input_image_end is not None:
        image2 = F.to_tensor(input_image_end) * 2 - 1
        masked_videos[0, -1] = image2.to(DEVICE)
        context_frame_masks[0, -1] = 1.0

    layer_masks = torch.zeros((1, LAYER_CAPACITY, 1, 1, HEIGHT, WIDTH), dtype=torch.bool)
    for layer_idx in range(LAYER_CAPACITY):
        if input_layer_masks[layer_idx] is not None:
            mask = F.to_tensor(input_layer_masks[layer_idx]) > 0.5
            layer_masks[0, layer_idx, 0] = mask
    layer_masks = layer_masks.to(DEVICE)
    motion_scores = torch.tensor([input_layer_scores], dtype=WEIGHT_DTYPE, device=DEVICE)

    # prepare motion scores condition
    motion_scores = motion_scores[:, :, None, None, None, None]
    motion_score_mask = layer_masks.to(dtype=motion_scores.dtype)
    context_score = motion_scores * motion_score_mask - torch.ones_like(motion_scores) * (1 - motion_score_mask)
    context_score = torch.max(context_score, dim=1).values  # reduce to [b, f, c, h, w]
    context_score = context_score.repeat(1, LENGTH, 3, 1, 1)

    # prepare trajectory condition
    sketch = torch.ones((1, LAYER_CAPACITY, LENGTH, 3, HEIGHT, WIDTH), dtype=WEIGHT_DTYPE)
    for layer_idx in range(LAYER_CAPACITY):
        sketch_path = input_layer_sketches[layer_idx]
        if sketch_path is not None:
            video_reader = decord.VideoReader(sketch_path)
            assert len(video_reader) == LENGTH, f"Input the length of sketch sequence should match the video length."
            video_frames = video_reader.get_batch(range(LENGTH)).asnumpy()
            sketch_values = [F.to_tensor(IMG_TRANSFORM(Image.fromarray(frame))) for frame in video_frames]
            sketch_values = torch.stack(sketch_values) * 2 - 1
            sketch[0, layer_idx] = sketch_values
    sketch = sketch.to(DEVICE)
    context_sketch = torch.min(sketch, dim=1).values  # reduce to [b, f, c, h, w]

    # prepare trajectory condition
    heatmap = torch.zeros((1, LAYER_CAPACITY, LENGTH, 3, HEIGHT, WIDTH), dtype=WEIGHT_DTYPE)
    heatmap[:, :, :, 0] -= 1
    trajectory = []
    traj_layer_index = []
    for layer_idx in range(LAYER_CAPACITY):
        tracking_points = input_layer_trajectories[layer_idx]
        for temp_track in tracking_points:
            if len(temp_track) > 1:
                x = [point[0] for point in temp_track]
                y = [point[1] for point in temp_track]
                t = np.linspace(0, 1, len(temp_track))
                fx = PchipInterpolator(t, x)
                fy = PchipInterpolator(t, y)
                t_new = np.linspace(0, 1, LENGTH)
                x_new = fx(t_new)
                y_new = fy(t_new)
                temp_traj = np.stack([x_new, y_new], axis=-1).astype(np.float32)
                trajectory.append(temp_traj)
                traj_layer_index.append(layer_idx)
            elif len(temp_track) == 1:
                trajectory.append(np.array(temp_track * LENGTH))
                traj_layer_index.append(layer_idx)
    trajectory = np.stack(trajectory)
    trajectory = np.transpose(trajectory, (1, 0, 2))
    traj_layer_index = np.array(traj_layer_index)
    heatmap = generate_gaussian_heatmap(trajectory, WIDTH, HEIGHT, traj_layer_index, LAYER_CAPACITY, offset=True)
    heatmap = rearrange(heatmap, "f n c h w -> (f n) c h w")
    graymap, offset = heatmap[:, :1], heatmap[:, 1:]
    graymap = graymap / 255.
    rad = torch.sqrt(offset[:, 0:1]**2 + offset[:, 1:2]**2)
    rad_max = torch.max(rad)
    epsilon = 1e-5
    offset = offset / (rad_max + epsilon)
    graymap = graymap * 2 - 1
    heatmap = torch.cat([graymap, offset], dim=1)
    heatmap = rearrange(heatmap, '(f n) c h w -> n f c h w', n=LAYER_CAPACITY)
    heatmap = heatmap[None]
    indices = torch.max(heatmap[:, :, :, 0:1], dim=1, keepdim=True).indices  # [b, 1, f, 1, h, w]
    expanded_indices = indices.expand(-1, -1, -1, 3, -1, -1)  # [b, 1, f, 3, h, w]
    context_trajectory = torch.gather(heatmap, 1, expanded_indices) # [b, 1, f, c, h, w]
    context_trajectory = context_trajectory.squeeze(1)  # reduce to [b, f, c, h, w]
    context_trajectory = context_trajectory.to(device=DEVICE, dtype=WEIGHT_DTYPE)

    context_frame_latents = vae.encode(rearrange(masked_videos, "b f c h w -> b c f h w"))[0].sample()
    context_frame_masks = rearrange(context_frame_masks, "b f c h w -> b c f h w")
    context_frame_masks = 1 - context_frame_masks   # we follow vace to indicate the masked area as 1, and the unmasked area as 0
    length, height, width = context_frame_masks.shape[2:]
    new_length = int((length + 3) // VAE_STRIDE[0])
    height = 2 * (int(height) // (VAE_STRIDE[1] * 2))
    width = 2 * (int(width) // (VAE_STRIDE[2] * 2))
    context_frame_masks = context_frame_masks[:, 0]
    context_frame_masks = context_frame_masks.view(
        -1, length, height, VAE_STRIDE[1], width, VAE_STRIDE[2]
    )
    context_frame_masks = context_frame_masks.permute(0, 3, 5, 1, 2, 4)
    context_frame_masks = context_frame_masks.reshape(
        -1, VAE_STRIDE[1] * VAE_STRIDE[2], length, height, width
    )
    context_frame_masks = interpolate(
        context_frame_masks, size=(new_length, height, width), mode='nearest-exact'
    )

    context_score_latents = vae.encode(rearrange(context_score, "b f c h w -> b c f h w"))[0].sample()
    context_sketch_latents = vae.encode(rearrange(context_sketch, "b f c h w -> b c f h w"))[0].sample()
    context_trajectory_latents = vae.encode(rearrange(context_trajectory, "b f c h w -> b c f h w"))[0].sample()

    context_control = torch.cat([context_frame_latents, context_frame_masks, context_score_latents, context_sketch_latents, context_trajectory_latents], dim=1)

    # Get the text embedding for conditioning
    text_embeddings = text_encoder([prompt], DEVICE)

    if do_classifier_free_guidance:
        un_text_embeddings = text_encoder([n_prompt], DEVICE)
        text_embeddings = un_text_embeddings + text_embeddings
        context_control = torch.cat([context_control] * 2, dim=0)

    latent_shape = context_frame_latents.shape
    latents = torch.randn(latent_shape, generator=generator, device=DEVICE, dtype=WEIGHT_DTYPE)
    seq_len = math.ceil((latent_shape[3] * latent_shape[4]) /
                        (PATCH_SIZE[1] * PATCH_SIZE[2]) * latent_shape[2])

    noise_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False
    )
    noise_scheduler.set_timesteps(
        num_inference_steps, device=DEVICE, shift=8.0
    )
    timesteps = noise_scheduler.timesteps
    for _, t in enumerate(tqdm(timesteps)):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        timestep = torch.tensor([t], device=DEVICE, dtype=torch.long)
        timestep = timestep.expand(latent_model_input.shape[0])

        with torch.amp.autocast('cuda', dtype=WEIGHT_DTYPE):
            noise_pred = video_model(
                latent_model_input,
                context=text_embeddings,
                t=timestep,
                seq_len=seq_len,
                vace_context=context_control,
                vace_context_scale=1.0,
            )

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = noise_scheduler.step(
            noise_pred,
            t,
            latents,
            return_dict=False,
            generator=generator,
        )[0]

    video = vae.decode(latents).sample
    video = (video / 2 + 0.5).clamp(0, 1)
    video = video.cpu().float()
    output_video_path = os.path.join(savedir, "video.mp4")
    save_videos_grid(video, output_video_path, fps=24)
    output_video_traj_path = os.path.join(savedir, "video_with_traj.mp4")
    save_videos_with_traj(video[0], torch.from_numpy(trajectory), output_video_traj_path, fps=24, line_width=7, circle_radius=10)
    return output_video_path, output_video_traj_path


if __name__ == "__main__":
    main()