import numpy as np
from pathlib import Path
from PIL import Image

from .compressor import BaseCompressor
from . import utils


class FourierCompressor(BaseCompressor):
    def __init__(self, keep_fraction: float = 0.1):
        """
        Compressor based on 2D Fast Fourier Transform (FFT), low-pass filter.

        Parameters:
        -----------
        - keep_fraction: float
              Fraction of central (low) frequencies to retain. Must be in (0, 1].
        """
        super().__init__()
        if not (0 < keep_fraction <= 1):
            raise ValueError("keep_fraction must be in (0, 1].")
        self.keep_fraction = keep_fraction

    def compress(self, input_image_path: str, output_dir: str) -> Image.Image:
        """
        Compress an RGB image by retaining only a central fraction of FFT frequencies.

        Parameters:
        -----------
        - input_image_path: str
              Path to the input image.
        - output_dir: str
              Directory where the compressed image will be saved.

        Returns:
        --------
        - compressed_image: PIL.Image.Image
              The compressed image.
        """
        _, img_np = utils.load_image(input_image_path)

        compressed_channels = [
            np.clip(self._compress_channel(img_np[:, :, i]), 0, 255).astype(np.uint8)
            for i in range(3)
        ]

        compressed_image = Image.fromarray(np.stack(compressed_channels, axis=2))

        filename = utils.define_name_for_compressed_image(
            input_image_path, "fourier", [int(self.keep_fraction * 100)]
        )
        utils.save_image(compressed_image, str(Path(output_dir) / filename))

        return compressed_image

    def _compress_channel(self, channel: np.ndarray) -> np.ndarray:
        """
        Apply FFT-based low-pass compression to a single image channel.

        Parameters:
        -----------
        - channel: np.ndarray, shape (H, W)
              Single-channel image data.

        Returns:
        --------
        - reconstructed: np.ndarray, shape (H, W), dtype float64
              Reconstructed channel after low-pass filtering.
        """
        fft_shifted = np.fft.fftshift(np.fft.fft2(channel))

        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        r_keep = int(rows * self.keep_fraction / 2)
        c_keep = int(cols * self.keep_fraction / 2)

        mask = np.zeros((rows, cols), dtype=bool)
        mask[crow - r_keep:crow + r_keep, ccol - c_keep:ccol + c_keep] = True

        fft_shifted[~mask] = 0

        return np.abs(np.fft.ifft2(np.fft.ifftshift(fft_shifted)))
