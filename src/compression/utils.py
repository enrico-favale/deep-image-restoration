from pathlib import Path
from PIL import Image
import numpy as np


def extract_filename_from_path(path: str) -> str:
    """
    Given a path to a file extracts the file name without extension.

    Parameters:
    -----------
    - path: str
          Path to file (includes the filename and extension).

    Returns:
    --------
    - name: str
          File name without extension.
    """
    return Path(path).stem


def define_name_for_compressed_image(
    input_image_path: str, method: str, parameters: list
) -> str:
    """
    Define the output filename for a compressed image based on method and parameters.

    Parameters:
    -----------
    - input_image_path: str
          Path to the original image. Used to extract the base filename.
    - method: str
          Compression method. Options: "fourier" | "wavelet" | "other".
    - parameters: list
          Ordered list of parameters used for compression, appended to the filename.
          Examples:
            Fourier  -> [0.1]           produces "rubik_fourier_0.1.png"
            Wavelet  -> ["haar", 1, 20] produces "rubik_wavelet_haar_1_20.png"

    Returns:
    --------
    - filename: str
          Output filename with .png extension.
    """
    base = extract_filename_from_path(input_image_path)
    parts = [base, method] + [str(p) for p in parameters]
    return "_".join(parts) + ".png"


def load_image(path: str) -> tuple[Image.Image, np.ndarray]:
    """
    Load an image from disk and convert it to a NumPy array.

    Parameters:
    -----------
    - path: str
        Path to the image file.

    Returns:
    --------
    - image: PIL.Image.Image
        The loaded PIL image.
    - image_np: np.ndarray, shape (H, W, 3), dtype uint8
        The image as a NumPy array in RGB format.
    """

    image = Image.open(path).convert("RGB")

    return image, np.array(image)


def save_image(image: Image.Image, path: str) -> None:
    """
    Save a PIL image to disk.

    Parameters:
    -----------
    - image: PIL.Image.Image
          The image to save.
    - path: str
          Destination path.
    """

    image.save(path)
