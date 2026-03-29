from pathlib import Path
from typing import Literal
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim


class ModelEvaluator:
    """
    Evaluates an image restoration model on a test DataLoader.

    Computes PSNR, SSIM, MSE, MAE, and optionally LPIPS for both the
    baseline (compressed vs original) and the restored output (model output
    vs original), exposing each metric as an independent method.

    Parameters
    ----------
    - model : torch.nn.Module, required
        PyTorch model to evaluate. Must return a tuple (reconstruction, latent).
    - test_loader : torch.utils.data.DataLoader, required
        DataLoader yielding (compressed, original) image pairs, values in [0, 1].
    - device : torch.device, optional
        Device used for inference. Defaults to CUDA if available, else CPU.
    - checkpoint : Path, optional
        Path to a .pth checkpoint file. If provided, weights are loaded before evaluation.
    - lpips_net : str, optional
        Backbone for LPIPS ('alex' or 'vgg'). If None, LPIPS is skipped.
        Requires `torchmetrics[image]` or `lpips` to be installed.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device = None,
        checkpoint: Path = None,
        lpips_net: Literal["alex", "vgg"] | None = "alex",
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.test_loader = test_loader

        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
            print(f"✔ Checkpoint loaded from {checkpoint}")

        self.model.eval()

        # LPIPS — opzionale, richiede torchmetrics[image]
        self._lpips_metric = None
        if lpips_net is not None:
            try:
                from torchmetrics.image.lpip import (
                    LearnedPerceptualImagePatchSimilarity,
                )

                self._lpips_metric = LearnedPerceptualImagePatchSimilarity(
                    net_type=lpips_net,
                    normalize=True,  # normalize=True accetta input in [0,1]
                ).to(self.device)
                print(f"✔ LPIPS initialized with backbone '{lpips_net}'")
            except ImportError:
                print("⚠ torchmetrics[image] not found — LPIPS will be skipped.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, verbose: bool = True) -> dict:
        """
        Run full evaluation: all metrics for baseline and restored images.

        Parameters
        ----------
        - verbose : bool, optional
            If True, prints a formatted summary. Defaults to True.

        Returns
        -------
        - results : dict
            Dictionary with keys: psnr_baseline, ssim_baseline, mse_baseline,
            mae_baseline, psnr_restored, ssim_restored, mse_restored, mae_restored,
            delta_psnr, delta_ssim, delta_mse, delta_mae.
            Also includes lpips_baseline, lpips_restored, delta_lpips if LPIPS
            was initialized successfully.
        """
        results = {
            "psnr_baseline": self.baseline_psnr(),
            "ssim_baseline": self.baseline_ssim(),
            "mse_baseline": self.baseline_mse(),
            "mae_baseline": self.baseline_mae(),
            "psnr_restored": self.restored_psnr(),
            "ssim_restored": self.restored_ssim(),
            "mse_restored": self.restored_mse(),
            "mae_restored": self.restored_mae(),
        }
        results["delta_psnr"] = results["psnr_restored"] - results["psnr_baseline"]
        results["delta_ssim"] = results["ssim_restored"] - results["ssim_baseline"]
        results["delta_mse"] = results["mse_restored"] - results["mse_baseline"]
        results["delta_mae"] = results["mae_restored"] - results["mae_baseline"]

        if self._lpips_metric is not None:
            results["lpips_baseline"] = self.baseline_lpips()
            results["lpips_restored"] = self.restored_lpips()
            results["delta_lpips"] = (
                results["lpips_restored"] - results["lpips_baseline"]
            )

        if verbose:
            self._print_results(results)

        return results

    # ------------------------------------------------------------------
    # Baseline metrics (compressed vs original)
    # ------------------------------------------------------------------

    def baseline_psnr(self) -> float:
        """
        Compute mean PSNR between compressed and original images (no model involved).

        Serves as a lower-bound reference: how much quality was lost by compression alone.

        Returns
        -------
        - psnr : float
            Mean PSNR in dB over the entire test set.
        """
        scores = []
        for compressed, original in self.test_loader:
            for c, o in zip(compressed.numpy(), original.numpy()):
                scores.append(
                    calc_psnr(
                        np.transpose(o, (1, 2, 0)),
                        np.transpose(c, (1, 2, 0)),
                        data_range=1.0,
                    )
                )
        return float(np.mean(scores))

    def baseline_ssim(self) -> float:
        """
        Compute mean SSIM between compressed and original images (no model involved).

        Returns
        -------
        - ssim : float
            Mean SSIM in [0, 1] over the entire test set.
        """
        scores = []
        for compressed, original in self.test_loader:
            for c, o in zip(compressed.numpy(), original.numpy()):
                scores.append(
                    calc_ssim(
                        np.transpose(o, (1, 2, 0)),
                        np.transpose(c, (1, 2, 0)),
                        data_range=1.0,
                        channel_axis=2,
                    )
                )
        return float(np.mean(scores))

    def baseline_mse(self) -> float:
        """
        Compute mean MSE between compressed and original images (no model involved).

        MSE is the average squared pixel-level difference. It is directly related
        to PSNR via PSNR = 10 * log10(1 / MSE) for data_range=1.

        Returns
        -------
        - mse : float
            Mean MSE over the entire test set.
        """
        scores = []
        for compressed, original in self.test_loader:
            diff = (compressed - original) ** 2
            scores.append(diff.mean().item())
        return float(np.mean(scores))

    def baseline_mae(self) -> float:
        """
        Compute mean MAE between compressed and original images (no model involved).

        MAE is more robust to outlier pixels than MSE, making it a useful
        complement when large localized artifacts are present.

        Returns
        -------
        - mae : float
            Mean MAE over the entire test set.
        """
        scores = []
        for compressed, original in self.test_loader:
            diff = (compressed - original).abs()
            scores.append(diff.mean().item())
        return float(np.mean(scores))

    def baseline_lpips(self) -> float:
        """
        Compute mean LPIPS between compressed and original images (no model involved).

        LPIPS measures perceptual distance using deep features from a pre-trained
        network (AlexNet or VGG). Lower values indicate greater perceptual similarity.
        Requires torchmetrics[image] and the lpips_net argument set at construction.

        Returns
        -------
        - lpips : float
            Mean LPIPS score over the entire test set. Returns -1.0 if LPIPS
            was not initialized.
        """
        if self._lpips_metric is None:
            return -1.0
        scores = []
        with torch.no_grad():
            for compressed, original in self.test_loader:
                compressed = compressed.to(self.device)
                original = original.to(self.device)
                scores.append(self._lpips_metric(compressed, original).item())
        return float(np.mean(scores))

    # ------------------------------------------------------------------
    # Restored metrics (model output vs original)
    # ------------------------------------------------------------------

    def restored_psnr(self) -> float:
        """
        Compute mean PSNR between the model's restored output and the original images.

        Returns
        -------
        - psnr : float
            Mean PSNR in dB over the entire test set.
        """
        scores = []
        with torch.no_grad():
            for compressed, original in self.test_loader:
                recon, _ = self.model(compressed.to(self.device))
                for r, o in zip(recon.cpu().numpy(), original.numpy()):
                    scores.append(
                        calc_psnr(
                            np.transpose(o, (1, 2, 0)),
                            np.transpose(r, (1, 2, 0)),
                            data_range=1.0,
                        )
                    )
        return float(np.mean(scores))

    def restored_ssim(self) -> float:
        """
        Compute mean SSIM between the model's restored output and the original images.

        Returns
        -------
        - ssim : float
            Mean SSIM in [0, 1] over the entire test set.
        """
        scores = []
        with torch.no_grad():
            for compressed, original in self.test_loader:
                recon, _ = self.model(compressed.to(self.device))
                for r, o in zip(recon.cpu().numpy(), original.numpy()):
                    scores.append(
                        calc_ssim(
                            np.transpose(o, (1, 2, 0)),
                            np.transpose(r, (1, 2, 0)),
                            data_range=1.0,
                            channel_axis=2,
                        )
                    )
        return float(np.mean(scores))

    def restored_mse(self) -> float:
        """
        Compute mean MSE between the model's restored output and the original images.

        Returns
        -------
        - mse : float
            Mean MSE over the entire test set.
        """
        scores = []
        with torch.no_grad():
            for compressed, original in self.test_loader:
                recon, _ = self.model(compressed.to(self.device))
                diff = (recon.cpu() - original) ** 2
                scores.append(diff.mean().item())
        return float(np.mean(scores))

    def restored_mae(self) -> float:
        """
        Compute mean MAE between the model's restored output and the original images.

        Returns
        -------
        - mae : float
            Mean MAE over the entire test set.
        """
        scores = []
        with torch.no_grad():
            for compressed, original in self.test_loader:
                recon, _ = self.model(compressed.to(self.device))
                diff = (recon.cpu() - original).abs()
                scores.append(diff.mean().item())
        return float(np.mean(scores))

    def restored_lpips(self) -> float:
        """
        Compute mean LPIPS between the model's restored output and the original images.

        LPIPS measures perceptual distance using deep features from a pre-trained
        network. A model that produces sharp but slightly shifted textures may have
        good PSNR/SSIM but poor LPIPS, revealing over-smoothing or hallucination.

        Returns
        -------
        - lpips : float
            Mean LPIPS score over the entire test set. Returns -1.0 if LPIPS
            was not initialized.
        """
        if self._lpips_metric is None:
            return -1.0
        scores = []
        with torch.no_grad():
            for compressed, original in self.test_loader:
                recon, _ = self.model(compressed.to(self.device))
                recon = recon.clamp(0.0, 1.0)
                scores.append(
                    self._lpips_metric(recon, original.to(self.device)).item()
                )
        return float(np.mean(scores))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _print_results(self, r: dict) -> None:
        sep = "─" * 54
        print(f"\n{sep}")
        print(f"  {'Metric':<18} {'Baseline':>10} {'Restored':>10} {'Delta':>10}")
        print(sep)
        print(
            f"  {'PSNR (dB)':<18} {r['psnr_baseline']:>10.2f} {r['psnr_restored']:>10.2f} {r['delta_psnr']:>+10.2f}"
        )
        print(
            f"  {'SSIM':<18} {r['ssim_baseline']:>10.4f} {r['ssim_restored']:>10.4f} {r['delta_ssim']:>+10.4f}"
        )
        print(
            f"  {'MSE':<18} {r['mse_baseline']:>10.5f} {r['mse_restored']:>10.5f} {r['delta_mse']:>+10.5f}"
        )
        print(
            f"  {'MAE':<18} {r['mae_baseline']:>10.5f} {r['mae_restored']:>10.5f} {r['delta_mae']:>+10.5f}"
        )
        if "lpips_restored" in r:
            print(
                f"  {'LPIPS ↓':<18} {r['lpips_baseline']:>10.4f} {r['lpips_restored']:>10.4f} {r['delta_lpips']:>+10.4f}"
            )
        print(f"{sep}\n")
