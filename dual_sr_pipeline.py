# File: dual_sr_pipeline.py

import numpy as np
import cv2
from skimage import io, img_as_float, img_as_ubyte, color
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
from scipy import ndimage
import torch
import torch.nn as nn
from piq import niqe

# -------------------------------
# 1. Image IO
# -------------------------------
def load_image_gray(path):
    img = io.imread(path, as_gray=True)
    return img_as_float(img)

def show_images(img_list, titles=None, cmap='gray'):
    n = len(img_list)
    plt.figure(figsize=(5 * n, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(img_list[i], cmap=cmap)
        if titles: plt.title(titles[i])
        plt.axis('off')
    plt.show()

# -------------------------------
# 2. Registration
# -------------------------------
def estimate_shift_phase_correlation(img1, img2):
    shift, _ = cv2.phaseCorrelate(np.float32(img1), np.float32(img2))
    print(f"[INFO] Estimated shift: x={shift[0]:.4f}, y={shift[1]:.4f}")
    return shift

def shift_image(img, shift):
    return ndimage.shift(img, shift[::-1], mode='reflect')

# -------------------------------
# 3. Fusion
# -------------------------------
def fuse_shift_and_add(img1, img2, shift, scale_factor=2):
    h, w = img1.shape
    new_h, new_w = h * scale_factor, w * scale_factor
    up1 = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    up2 = cv2.resize(img2, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    hr_shift = (shift[0] * scale_factor, shift[1] * scale_factor)
    print(f"[INFO] HR grid shift: x={hr_shift[0]:.4f}, y={hr_shift[1]:.4f}")
    up2_shifted = ndimage.shift(up2, hr_shift[::-1], mode='reflect')
    return (up1 + up2_shifted) / 2.0

# -------------------------------
# 4. DL Super-Resolution
# -------------------------------
class DualImageFusionSRNet(nn.Module):
    def __init__(self, num_channels=1, upscale_factor=2):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU()
        )
        self.fusion = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64 * (upscale_factor ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU(),
            nn.Conv2d(64, num_channels, 3, 1, 1)
        )

    def forward(self, x1, x2):
        f1 = self.encoder1(x1)
        f2 = self.encoder2(x2)
        fused = self.fusion(torch.cat([f1, f2], dim=1))
        return self.decoder(fused)

def load_pretrained_model(model_path=None, device='cpu'):
    model = DualImageFusionSRNet()
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("[INFO] Loaded pretrained weights!")
    return model.to(device).eval()

def predict_sr(model, img1, img2, device='cpu'):
    with torch.no_grad():
        t1 = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0).float().to(device)
        t2 = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).float().to(device)
        out = model(t1, t2).squeeze().cpu().numpy()
        return np.clip(out, 0, 1)

# -------------------------------
# 5. Blind IQA
# -------------------------------
def compute_niqe_score(image):
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    img_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()
    try:
        return niqe(img_tensor, data_range=1.0).item()
    except Exception as e:
        print("[WARN] NIQE failed:", e)
        return None

def compute_entropy(image):
    return shannon_entropy(image)

def generate_niqe_heatmap(image, patch_size=64, stride=32):
    h, w = image.shape
    heatmap = np.zeros((h, w))
    counts = np.zeros((h, w))
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            score = compute_niqe_score(patch)
            if score is not None:
                heatmap[y:y+patch_size, x:x+patch_size] += score
                counts[y:y+patch_size, x:x+patch_size] += 1
    counts[counts == 0] = 1
    return heatmap / counts

def show_heatmap(heatmap, title="NIQE Heatmap"):
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap='inferno')
    plt.colorbar(label="NIQE Score (lower = better)")
    plt.title(title)
    plt.axis('off')
    plt.show()

# -------------------------------
# 6. Pipeline Runner
# -------------------------------
def run_pipeline(path1, path2, model_path=None, device='cpu'):
    img1 = load_image_gray(path1)
    img2 = load_image_gray(path2)

    shift = estimate_shift_phase_correlation(img1, img2)
    img2_aligned = shift_image(img2, shift)
    fused = fuse_shift_and_add(img1, img2, shift)

    model = load_pretrained_model(model_path, device)
    sr_img = predict_sr(model, img1, img2_aligned, device)

    niqe_score = compute_niqe_score(sr_img)
    entropy_score = compute_entropy(sr_img)
    heatmap = generate_niqe_heatmap(sr_img)

    show_images([img1, img2, img2_aligned], ["LR1", "LR2", "Aligned LR2"])
    show_images([fused, sr_img], ["Fused Image", "SR Output"])
    show_heatmap(heatmap)

    return {
        "super_resolved": sr_img,
        "niqe": niqe_score,
        "entropy": entropy_score,
        "heatmap": heatmap
    }
