import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

def plot_restoration(
    original_path: str,
    compressed_path: str,
    restored_path: str,
    figsize: tuple = (15, 5),
) -> None:
    """
    Plots original / compressed / restored side by side with PSNR and SSIM.

    Parameters
    ----------
    - original_path : str, required
        path to the original (ground truth) image
    - compressed_path : str, required
        path to the Fourier-compressed image
    - restored_path : str, required
        path to the restored path.
    - figsize : tuple, optional
        matplotlib figure size
    """

    # ── Load images ──────────────────────────────────────────────────────────
    original = (
        np.array(Image.open(original_path).convert("RGB").resize((256, 256))) / 255.0
    )
    compressed = (
        np.array(Image.open(compressed_path).convert("RGB").resize((256, 256))) / 255.0
    )
    restored = (
        np.array(Image.open(restored_path).convert("RGB").resize((256, 256))) / 255.0
    )

    # ── Metrics ──────────────────────────────────────────────────────────────
    psnr_before = calc_psnr(original, compressed, data_range=1.0)
    ssim_before = calc_ssim(original, compressed, data_range=1.0, channel_axis=2)
    psnr_after = calc_psnr(original, restored, data_range=1.0)
    ssim_after = calc_ssim(original, restored, data_range=1.0, channel_axis=2)

    # ── Plot ─────────────────────────────────────────────────────────────────
    images = [original, compressed, restored]
    titles = [
        "Original",
        f"Compressed\nPSNR {psnr_before:.2f} dB  |  SSIM {ssim_before:.4f}",
        f"Restored\nPSNR {psnr_after:.2f} dB  |  SSIM {ssim_after:.4f}",
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.suptitle(
        f"PSNR {psnr_before:.2f} -> {psnr_after:.2f} dB  "
        f"(+{psnr_after - psnr_before:.2f})  |  "
        f"SSIM {ssim_before:.4f} -> {ssim_after:.4f}  "
        f"(+{ssim_after - ssim_before:.4f})",
        fontsize=10,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()
