from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim


def make_output_name(
    compressed_path: Path,
    model_path: Path,
    model_labels: dict[Path, str] | None = None,
) -> str:
    """
    Builds the output filename for a restored image.

    Parameters
    ----------
    - compressed_path : Path
        Path to the compressed image.
    - model_path : Path
        Path to the model checkpoint.
    - model_labels : dict[Path, str] | None
        Optional mapping {model_path: label}. If None or key missing,
        falls back to model_path.stem.
    """
    compressed_stem = compressed_path.stem
    if model_labels is not None:
        model_label = model_labels.get(model_path, model_path.stem)
    else:
        model_label = model_path.stem
    return f"restored_{compressed_stem}_{model_label}.png"


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

    # Load images
    original = (
        np.array(Image.open(original_path).convert("RGB").resize((256, 256))) / 255.0
    )
    compressed = (
        np.array(Image.open(compressed_path).convert("RGB").resize((256, 256))) / 255.0
    )
    restored = (
        np.array(Image.open(restored_path).convert("RGB").resize((256, 256))) / 255.0
    )

    # Metrics
    psnr_before = calc_psnr(original, compressed, data_range=1.0)
    ssim_before = calc_ssim(original, compressed, data_range=1.0, channel_axis=2)
    psnr_after = calc_psnr(original, restored, data_range=1.0)
    ssim_after = calc_ssim(original, restored, data_range=1.0, channel_axis=2)

    # Plot
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


def plot_restoration_comparation_between_models(
    original_path: Path,
    compressions: list[dict],
    model_paths: list[Path],
    model_display_labels: list[str] | None = None,
    model_save_labels: dict[Path, str] | None = None,
    figsize_per_cell: tuple = (4, 4),
) -> None:
    """
    Plots a grid comparing restorations across compressions and models.

    Layout:
        Row 0 : [Original] [Compressed_0] [Restored by model_1] ... [Restored by model_N]
        Row i : [  empty ] [Compressed_i] [Restored by model_1] ... [Restored by model_N]

    Parameters
    ----------
    - original_path : Path, required
        Path to the original (ground truth) image.
    - compressions : list[dict], required
        Each dict must have:
            - "compressed_path" : Path  -> compressed image
            - "label"           : str   -> row label (e.g. "Fourier keep=0.25")
    - model_paths : list[Path], required
        Ordered list of checkpoint paths used to restore each compressed image.
    - model_display_labels : list[str] | None, optional
        Human-readable labels shown in plot titles for each model.
        Must have the same length as model_paths. Defaults to model_path.stem.
    - model_save_labels : dict[Path, str] | None, optional
        Mapping {model_path: save_label} used to reconstruct restored image
        filenames (must match the labels used during saving via make_output_name).
        If None, falls back to model_path.stem for each model.
    - figsize_per_cell : tuple, optional
        (width, height) in inches per cell. Total figure size scales
        automatically with the number of models and compressions.
    """

    if model_display_labels is None:
        model_display_labels = [p.stem for p in model_paths]

    assert len(model_display_labels) == len(model_paths)

    model_labels_dict: dict[Path, str] = model_save_labels or {}

    n_models = len(model_paths)
    n_comps  = len(compressions)
    n_cols   = 1 + n_models      # col 0: original/compressed | cols 1..N: restored
    n_rows   = 1 + n_comps       # row 0: original | rows 1..M: compressions
    figsize  = (figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows)

    def load(path: Path) -> np.ndarray:
        return np.array(Image.open(path).convert("RGB").resize((256, 256))) / 255.0

    def metrics(ref: np.ndarray, img: np.ndarray) -> tuple[float, float]:
        psnr = calc_psnr(ref, img, data_range=1.0)
        ssim = calc_ssim(ref, img, data_range=1.0, channel_axis=2)
        return psnr, ssim

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    original = load(original_path)

    # Row 0: original in col 0, rest hidden
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original\n(ground truth)", fontsize=8)
    axes[0, 0].axis("off")
    for col_idx in range(1, n_cols):
        axes[0, col_idx].axis("off")

    # Rows 1..M: one per compression
    for row_idx, comp in enumerate(compressions, start=1):
        compressed_path = comp["compressed_path"]
        row_label       = comp.get("label", compressed_path.stem)

        compressed           = load(compressed_path)
        psnr_comp, ssim_comp = metrics(original, compressed)

        # Col 0: compressed
        axes[row_idx, 0].imshow(compressed)
        axes[row_idx, 0].set_title(
            f"[{row_label}]\nPSNR {psnr_comp:.2f} dB  |  SSIM {ssim_comp:.4f}",
            fontsize=8,
        )
        axes[row_idx, 0].axis("off")

        # Cols 1..N: restored
        restored_metrics_row: list[tuple[float, float]] = []
        for col_idx, (model_path, label) in enumerate(
            zip(model_paths, model_display_labels), start=1
        ):
            out_name = make_output_name(compressed_path, model_path, model_labels_dict)
            restored = load(compressed_path.parent / out_name)
            psnr_r, ssim_r = metrics(original, restored)
            restored_metrics_row.append((psnr_r, ssim_r))

            axes[row_idx, col_idx].imshow(restored)
            axes[row_idx, col_idx].set_title(
                f"{label}\n"
                f"PSNR {psnr_r:.2f} dB  (+{psnr_r - psnr_comp:.2f})  |  "
                f"SSIM {ssim_r:.4f}  (+{ssim_r - ssim_comp:.4f})",
                fontsize=8,
            )
            axes[row_idx, col_idx].axis("off")

    plt.suptitle(
        "Restoration comparison across compressions and models",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.show()
