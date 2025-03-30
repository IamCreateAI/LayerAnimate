import argparse
import sys
import datetime
import os
from omegaconf import OmegaConf

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

import diffusers
from diffusers import DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from lvdm.models.unet import UNetModel
from lvdm.models.autoencoder import AutoencoderKL, AutoencoderKL_Dualref
from lvdm.models.condition import FrozenOpenCLIPEmbedder, FrozenOpenCLIPImageEmbedderV2, Resampler
from lvdm.models.layer_controlnet import LayerControlNet
from lvdm.pipelines.pipeline_animation import AnimationPipeline
from lvdm.utils import generate_gaussian_heatmap, save_videos_grid, save_videos_with_traj

from einops import rearrange
import decord
from pathlib import Path
from PIL import Image
import numpy as np

# import debugpy
# debugpy.listen(5678)
# print("Waiting for debugger attach")
# debugpy.wait_for_client()

@torch.no_grad()
def main(args):
    if args.savedir is None:
        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        savedir = f"samples/{Path(args.config).stem}-{time_str}"
    else:
        savedir = args.savedir
    os.makedirs(savedir, exist_ok=True)

    config  = OmegaConf.load(args.config)
    weight_dtype = torch.bfloat16 if config["mixed_precision"] == "bf16" else torch.float32
    mode = config.get("mode", "interpolate")
    # create validation pipeline
    scheduler         = DDIMScheduler.from_pretrained(config.pretrained_model_path, subfolder="scheduler")
    text_encoder      = FrozenOpenCLIPEmbedder().eval()
    image_encoder     = FrozenOpenCLIPImageEmbedderV2().eval()
    image_projector   = Resampler.from_pretrained(config.pretrained_model_path, subfolder="image_projector").eval()
    vae, vae_dualref = None, None
    if mode == "interpolate":
        vae_dualref   = AutoencoderKL_Dualref.from_pretrained(config.pretrained_model_path, subfolder="vae_dualref").eval()
    else:
        vae           = AutoencoderKL.from_pretrained(config.pretrained_model_path, subfolder="vae").eval()
    unet              = UNetModel.from_pretrained(config.pretrained_model_path, subfolder="unet").eval()
    layer_controlnet  = LayerControlNet.from_pretrained(config.pretrained_model_path, subfolder="layer_controlnet").eval()

    pipeline = AnimationPipeline(
        vae=vae, vae_dualref=vae_dualref, text_encoder=text_encoder, image_encoder=image_encoder, image_projector=image_projector,
        unet=unet, layer_controlnet=layer_controlnet,
        scheduler=scheduler,
    ).to(device=args.device, dtype=weight_dtype)
    if mode == "interpolate":
        pipeline.vae_dualref.decoder.to(dtype=torch.float32)
    if config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            pipeline.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if config.seed is None:
        generator = None
    else:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        generator = torch.Generator(args.device).manual_seed(config.seed)

    config.W = config.get("W", args.W)
    config.H = config.get("H", args.H)
    config.L = config.get("L", args.L)

    image_transforms = transforms.Compose([
        transforms.Resize(min(config.H, config.W)),
        transforms.CenterCrop((config.H, config.W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    mask_transforms = transforms.Compose([
        transforms.Resize(min(config.H, config.W)),
        transforms.CenterCrop((config.H, config.W)),
    ])

    demo_dir = config.get("demo_dir")
    first_frame = os.path.join(demo_dir, "first_frame.jpg")
    if mode == "interpolate":
        last_frame = os.path.join(demo_dir, "last_frame.jpg")
    else:
        last_frame = None
    sketch_path = os.path.join(demo_dir, "sketch.mp4")
    trajectory_path = os.path.join(demo_dir, "trajectory.npz")

    if last_frame is None:
        # Image to Video
        image = image_transforms(Image.open(first_frame).convert("RGB"))
        frame_tensor = image[None].to(args.device)  # [F, C, H, W]
    else:
        # Interpolate
        image1 = image_transforms(Image.open(first_frame).convert("RGB"))
        image2 = image_transforms(Image.open(last_frame).convert("RGB"))
        frame_tensor1 = image1[None]
        frame_tensor2 = image2[None]
        frame_tensor = torch.cat([frame_tensor1, frame_tensor2], dim=0).to(args.device)
    frame_tensor = frame_tensor[None]

    if mode == "interpolate":
        layer_masks = torch.zeros((1, config.layer_capacity, 2, 1, config.H, config.W), dtype=torch.bool)
    else:
        layer_masks = torch.zeros((1, config.layer_capacity, 1, 1, config.H, config.W), dtype=torch.bool)
    for layer_idx in range(config.layer_capacity):
        mask_path = os.path.join(demo_dir, f"layer_{layer_idx}.jpg")
        if os.path.exists(mask_path):
            mask = mask_transforms(Image.open(mask_path).convert("L"))
            mask = F.to_tensor(mask) > 0.5
            layer_masks[0, layer_idx, 0] = mask
        last_mask_path = os.path.join(demo_dir, f"layer_{layer_idx}_last.jpg")
        if os.path.exists(last_mask_path) and mode == "interpolate":
            mask = mask_transforms(Image.open(last_mask_path).convert("L"))
            mask = F.to_tensor(mask) > 0.5
            layer_masks[0, layer_idx, 1] = mask
    layer_masks = layer_masks.to(args.device)
    layer_regions = layer_masks * frame_tensor[:, None]
    layer_validity = torch.tensor([config.layer_validity], dtype=torch.bool, device=args.device)
    motion_scores = torch.tensor([config.motion_scores], dtype=weight_dtype, device=args.device)
    layer_static = torch.tensor([config.layer_static], dtype=torch.bool, device=args.device)

    sketch = torch.ones((1, config.layer_capacity, config.L, 3, config.H, config.W), dtype=weight_dtype)
    if os.path.exists(sketch_path):
        video_reader = decord.VideoReader(sketch_path)
        assert len(video_reader) == config.L, f"Input the length of sketch sequence should match the video length."
        video_frames = video_reader.get_batch(range(config.L)).asnumpy()
        sketch_values = [image_transforms(Image.fromarray(frame)) for frame in video_frames]
        sketch_values = torch.stack(sketch_values, dim=0)
        sketch[0, config.sketch_layer_index] = sketch_values
    sketch = sketch.to(args.device)

    heatmap = torch.zeros((1, config.layer_capacity, config.L, 3, config.H, config.W), dtype=weight_dtype)
    heatmap[:, :, :, 0] -= 1
    if os.path.exists(trajectory_path):
        traj_file = np.load(trajectory_path)
        traj_width = traj_file["width"]
        traj_height = traj_file["height"]
        trajectory = traj_file["trajectory"]
        if traj_width < traj_height:
            scale = min(config.H, config.W) / traj_width
            new_h = int(traj_height * scale)
            new_w = min(config.H, config.W)
        else:
            scale = min(config.H, config.W) / traj_height
            new_w = int(traj_width * scale)
            new_h = min(config.H, config.W)
        trajectory[..., :2] *= scale
        crop_x = int(round((new_w - config.W) / 2.0))
        crop_y = int(round((new_h - config.H) / 2.0))
        trajectory[..., 0] -= crop_x
        trajectory[..., 1] -= crop_y
        traj_layer_index = torch.zeros(trajectory.shape[1], dtype=torch.long) + config.traj_layer_index

        heatmap = generate_gaussian_heatmap(trajectory, config.W, config.H, traj_layer_index, config.layer_capacity, offset=True)
        heatmap = rearrange(heatmap, "f n c h w -> (f n) c h w")
        graymap, offset = heatmap[:, :1], heatmap[:, 1:]
        graymap = graymap / 255.
        rad = torch.sqrt(offset[:, 0:1]**2 + offset[:, 1:2]**2)
        rad_max = torch.max(rad)
        epsilon = 1e-5
        offset = offset / (rad_max + epsilon)
        graymap = graymap * 2 - 1
        heatmap = torch.cat([graymap, offset], dim=1)
        heatmap = mask_transforms(heatmap)  # no need for normalization
        heatmap = rearrange(heatmap, '(f n) c h w -> n f c h w', n=config.layer_capacity)
        heatmap = heatmap[None]
    heatmap = heatmap.to(args.device)

    sample = pipeline(
        config.prompt,
        config.L,
        config.H,
        config.W,
        frame_tensor,
        layer_masks             = layer_masks,
        layer_regions           = layer_regions,
        layer_static            = layer_static,
        motion_scores           = motion_scores,
        sketch                  = sketch,
        trajectory              = heatmap,
        layer_validity          = layer_validity,
        num_inference_steps     = config.num_inference_steps,
        guidance_scale          = config.guidance_scale,
        guidance_rescale        = config.guidance_rescale,
        negative_prompt         = config.n_prompt,
        num_videos_per_prompt   = config.num_videos_per_prompt,
        eta                     = config.eta,
        generator               = generator,
        fps                     = config.fps,
        mode                    = mode,
        weight_dtype            = weight_dtype,
        output_type             = "tensor",
    ).videos

    for idx, video in enumerate(sample):
        save_videos_grid(video[None], os.path.join(savedir, f"video_{idx}.mp4"), fps=8)
        print(f"Saved {os.path.join(savedir, f'video_{idx}.mp4')}")
        if os.path.exists(trajectory_path):
            save_videos_with_traj(video, torch.from_numpy(trajectory), os.path.join(savedir, f"video_{idx}_with_traj.mp4"), fps=8, line_width=7, circle_radius=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--savedir", type=str, default=None)

    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=320)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    main(args)