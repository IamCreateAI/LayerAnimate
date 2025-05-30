import numpy as np
import cv2
import torch
import os
from einops import rearrange
import imageio
import torchvision


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


def save_videos_with_traj(videos: torch.Tensor, trajectory: torch.Tensor, path: str, rescale=False, fps=8, line_width=7, circle_radius=10):
    # videos: [C, F, H, W]
    # trajectory: [F, N, 2]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    videos = rearrange(videos, "c f h w -> f h w c")
    if rescale:
        videos = (videos + 1) / 2
    videos = (videos * 255).numpy().astype(np.uint8)
    outputs = []
    for frame_idx, img in enumerate(videos):
        # img: [H, W, C], traj: [N, 2]
        # draw trajectory use cv2.line
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for traj_idx in range(trajectory.shape[1]):
            for history_idx in range(frame_idx):
                cv2.line(img, tuple(trajectory[history_idx, traj_idx].int().tolist()), tuple(trajectory[history_idx+1, traj_idx].int().tolist()), (0, 0, 255), line_width)
            cv2.circle(img, tuple(trajectory[frame_idx, traj_idx].int().tolist()), circle_radius, (100, 230, 160), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        outputs.append(img)
    imageio.mimsave(path, outputs, fps=fps)


def generate_gaussian_template(imgSize=200):
    """ Adapted from DragAnything: https://github.com/showlab/DragAnything/blob/79355363218a7eb9b3437a31b8604b6d436d9337/dataset/dataset.py#L110"""
    circle_img = np.zeros((imgSize, imgSize), np.float32)
    circle_mask = cv2.circle(circle_img, (imgSize//2, imgSize//2), imgSize//2, 1, -1)

    isotropicGrayscaleImage = np.zeros((imgSize, imgSize), np.float32)

    # Guass Map
    for i in range(imgSize):
        for j in range(imgSize):
            isotropicGrayscaleImage[i, j] = 1 / 2 / np.pi / (40 ** 2) * np.exp(
                -1 / 2 * ((i - imgSize / 2) ** 2 / (40 ** 2) + (j - imgSize / 2) ** 2 / (40 ** 2)))

    isotropicGrayscaleImage = isotropicGrayscaleImage * circle_mask
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)).astype(np.float32)
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)*255).astype(np.uint8)

    # isotropicGrayscaleImage = cv2.resize(isotropicGrayscaleImage, (40, 40))
    return isotropicGrayscaleImage


def generate_gaussian_heatmap(tracks, width, height, layer_index, layer_capacity, side=20, offset=True):
    heatmap_template = generate_gaussian_template()
    num_frames, num_points = tracks.shape[:2]
    if isinstance(tracks, torch.Tensor):
        tracks = tracks.cpu().numpy()
    if offset:
        offset_kernel = cv2.resize(heatmap_template / 255, (2 * side + 1, 2 * side + 1))
        offset_kernel /= np.sum(offset_kernel)
        offset_kernel /= offset_kernel[side, side]
    heatmaps = []
    for frame_idx in range(num_frames):
        if offset:
            layer_imgs = np.zeros((layer_capacity, height, width, 3), dtype=np.float32)
        else:
            layer_imgs = np.zeros((layer_capacity, height, width, 1), dtype=np.float32)
        layer_heatmaps = []
        for point_idx in range(num_points):
            x, y = tracks[frame_idx, point_idx]
            layer_id = layer_index[point_idx]
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            x1 = int(max(x - side, 0))
            x2 = int(min(x + side, width - 1))
            y1 = int(max(y - side, 0))
            y2 = int(min(y + side, height - 1))
            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue
            temp_map = cv2.resize(heatmap_template, (x2-x1, y2-y1))
            layer_imgs[layer_id, y1:y2,x1:x2, 0] = np.maximum(layer_imgs[layer_id, y1:y2,x1:x2, 0], temp_map)
            if offset:
                if frame_idx < (num_frames - 1):
                    next_x, next_y = tracks[frame_idx + 1, point_idx]
                else:
                    next_x, next_y = x, y
                layer_imgs[layer_id, int(y), int(x), 1] = next_x - x
                layer_imgs[layer_id, int(y), int(x), 2] = next_y - y
        for img in layer_imgs:
            if offset:
                img[:, :, 1:] = cv2.filter2D(img[:, :, 1:], -1, offset_kernel)
            else:
                img = cv2.cvtColor(img[:, :, 0].astype(np.uint8), cv2.COLOR_GRAY2RGB)
            layer_heatmaps.append(img)
        heatmaps.append(np.stack(layer_heatmaps, axis=0))
    heatmaps = np.stack(heatmaps, axis=0)
    return torch.from_numpy(heatmaps).permute(0, 1, 4, 2, 3).contiguous().float()   # [F, N_layer, C, H, W]
