import torch
import torch.nn as nn
from src.utils.logger import Logger


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for image restoration.

    U-Net-style architecture with skip connections between encoder
    and decoder. Takes a compressed image as input and reconstructs
    a clean approximation of the original.

    Parameters
    ---------------
    - base_channels: int, optional
        Base channel count used to scale internal feature maps
        (multiplied by 1, 2, 4, 8 across encoder stages).
    - logger: Logger, optional
        Logger instance to receive architecture metadata.
        If provided, architecture info is logged upon initialization.
    """

    def __init__(self, base_channels: int = 32, logger: Logger = None):
        super().__init__()
        self.base_channels = base_channels

        self.enc1 = self._block(3, self.base_channels)
        self.enc2 = self._block(self.base_channels, self.base_channels * 2)
        self.enc3 = self._block(self.base_channels * 2, self.base_channels * 4)
        self.enc4 = self._block(self.base_channels * 4, self.base_channels * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.base_channels * 8, self.base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(self.base_channels * 8),
            nn.ReLU(inplace=True),
        )

        self.dec4 = self._up_block(
            self.base_channels * 8 + self.base_channels * 8, self.base_channels * 4
        )
        self.dec3 = self._up_block(
            self.base_channels * 4 + self.base_channels * 4, self.base_channels * 2
        )
        self.dec2 = self._up_block(
            self.base_channels * 2 + self.base_channels * 2, self.base_channels
        )
        self.dec1 = self._up_block(
            self.base_channels + self.base_channels, self.base_channels
        )

        self.final = nn.Sequential(nn.Conv2d(self.base_channels, 3, 1), nn.Sigmoid())

        if logger is not None:
            logger.log_architecture(
                model_name=self.__class__.__name__,
                model_summary=self._build_summary(),
            )

    def _build_summary(self) -> str:
        """
        Build a human-readable summary string of the architecture.

        Returns
        ----------
        - summary: str
            Multi-line string describing architecture and parameter count.
        """

        lines = [
            str(self),
            "",
            f"base_channels : {self.base_channels}",
            f"encoder stages : 4  ({self.base_channels} -> {self.base_channels*2} -> {self.base_channels*4} -> {self.base_channels*8})",
            f"bottleneck : {self.base_channels*8} channels",
            f"decoder stages : 4  (with skip connections)",
            f"total params : {self.count_parameters():,}",
        ]

        return "\n".join(lines)

    def _block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """
        Build a strided convolution encoder block.

        Parameters
        ---------------
        - in_ch: int, required
            Number of input channels.
        - out_ch: int, required
            Number of output channels.

        Returns
        ----------
        - block: nn.Sequential
            Sequential block: Conv2d (stride=2) -> BatchNorm2d -> ReLU.
        """

        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _up_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """
        Build a transposed convolution decoder block.

        Parameters
        ---------------
        - in_ch: int, required
            Number of input channels (includes skip connection channels).
        - out_ch: int, required
            Number of output channels.

        Returns
        ----------
        - block: nn.Sequential
            Sequential block: ConvTranspose2d (stride=2) -> BatchNorm2d -> ReLU.
        """

        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder, bottleneck, and decoder.

        Parameters
        ---------------
        - x: torch.Tensor, required
            Compressed input image tensor.
            Shape: (B, 3, H, W), values in [0.0, 1.0].

        Returns
        ----------
        - out: torch.Tensor
            Reconstructed image produced by the autoencoder.
            Shape: (B, 3, H, W), values in [0.0, 1.0].
        - z: torch.Tensor
            Latent representation from the bottleneck.
            Shape: (B, base_channels*8, H/16, W/16).
        """

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        z = self.bottleneck(e4)

        d = self.dec4(torch.cat([z, e4], dim=1))
        d = self.dec3(torch.cat([d, e3], dim=1))
        d = self.dec2(torch.cat([d, e2], dim=1))
        d = self.dec1(torch.cat([d, e1], dim=1))

        out = self.final(d)

        return out, z

    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the model.

        Returns
        ----------
        - count: int
            Total number of learnable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
