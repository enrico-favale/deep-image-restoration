from abc import ABC, abstractmethod
from PIL import Image


class BaseCompressor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compress(self, input_image_path: str, output_image_path: str) -> Image.Image:
        """
        Compress an image and save the result to disk.

        Parameters:
        -----------
        - input_image_path: str
              Path to the input image.
        - output_image_path: str
              Path where the compressed image will be saved.

        Returns:
        --------
        - compressed_image: PIL.Image.Image
              The compressed image.
        """
        
        pass
