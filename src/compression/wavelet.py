import numpy as np
import pywt
from pathlib import Path
from PIL import Image

from .compressor import BaseCompressor
from . import utils


class WaveletCompressor(BaseCompressor):
    def __init__(
        self,
        wavelet: str = "haar",
        level: int = 1,
        threshold: float = 20.0,
        keep_ll_only: bool = True,
    ):
        """
        Compressor based on Discrete Wavelet Transform (DWT) with soft thresholding.

        Parameters:
        -----------
        - wavelet: str
            Wavelet type to use (e.g. "haar", "db2", "db4", "bior1.3", "coif1").
            Any wavelet supported by pywt.wavedec2 is valid.
        - level: int
            Decomposition level. Higher values increase compression but lose more detail.
        - threshold: float
            Soft thresholding value applied to wavelet coefficients.
        - keep_ll_only: bool
            If True, zeroes out all detail sub-bands (LH, HL, HH), retaining only
                the approximation (LL). Produces maximum smoothing.
            If False, applies soft thresholding to detail sub-bands instead.
        """
        super().__init__()

        if wavelet not in pywt.wavelist(kind="discrete"):
            raise ValueError(
                f"'{wavelet}' is not a valid discrete wavelet. See pywt.wavelist(kind='discrete')."
            )
        if level < 1:
            raise ValueError("level must be >= 1.")
        if threshold < 0:
            raise ValueError("threshold must be non-negative.")

        self.wavelet = wavelet
        self.level = level
        self.threshold = threshold
        self.keep_ll_only = keep_ll_only

    def compress(self, input_image_path: str, output_dir: str) -> Image.Image:
        """
        Compress an RGB image by applying DWT-based compression on each channel.

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
            self._compress_channel(img_np[:, :, i]).astype(np.uint8)
            for i in range(3)
        ]

        compressed_image = Image.fromarray(np.stack(compressed_channels, axis=2))

        parameters = [self.wavelet, self.level, int(self.threshold)]
        if self.keep_ll_only:
            parameters.append("ll_only")

        filename = utils.define_name_for_compressed_image(input_image_path, "wavelet", parameters)
        utils.save_image(compressed_image, str(Path(output_dir) / filename))

        return compressed_image

    def _compress_channel(self, channel: np.ndarray) -> np.ndarray:
        """
        Apply DWT-based compression to a single image channel.

        Parameters:
        -----------
        - channel: np.ndarray, shape (H, W)
              Single-channel image data.

        Returns:
        --------
        - reconstructed: np.ndarray, shape (H, W), dtype float64
              Reconstructed channel after wavelet compression, clipped to [0, 255].
        """

        coeffs = pywt.wavedec2(channel, wavelet=self.wavelet, level=self.level)

        compressed = [pywt.threshold(coeffs[0], self.threshold, mode="soft")]

        for detail_tuple in coeffs[1:]:
            if self.keep_ll_only:
                compressed.append(tuple(np.zeros_like(d) for d in detail_tuple))
            else:
                compressed.append(
                    tuple(
                        pywt.threshold(d, self.threshold, mode="soft")
                        for d in detail_tuple
                    )
                )

        reconstructed = pywt.waverec2(compressed, wavelet=self.wavelet)

        return np.clip(reconstructed, 0, 255)
