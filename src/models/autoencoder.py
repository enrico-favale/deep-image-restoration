import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, base_channels: int = 32):
        """
        Convolutional Autoencoder per image restoration.
        Input:  compressed image (3, H, W)
        Output: reconstructed image - ground truth approximation (3, H, W)

        Parameters
        ----------
        - base_channes : int, optional
            Number used to create internal architecture (multiplied by: 1, 2, 4, 8)

        """
        
        super().__init__()
        c = base_channels

        self.enc1 = self._block(3, c)
        self.enc2 = self._block(c, c * 2)
        self.enc3 = self._block(c * 2, c * 4)
        self.enc4 = self._block(c * 4, c * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(c * 8, c * 8, 3, padding=1),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
        )

        self.dec4 = self._up_block(c * 8 + c * 8, c * 4)
        self.dec3 = self._up_block(c * 4 + c * 4, c * 2)
        self.dec2 = self._up_block(c * 2 + c * 2, c)
        self.dec1 = self._up_block(c + c, c)

        self.final = nn.Sequential(nn.Conv2d(c, 3, 1), nn.Sigmoid())

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder, bottleneck and decoder.

        Parameters
        ----------
        - x : torch.Tensor, required
            Input tensor representing the compressed image.
            Shape: (B, 3, H, W), values in [0.0, 1.0].

        Returns
        -------
        - out : torch.Tensor
            Reconstructed image produced by the autoencoder.
            Shape: (B, 3, H, W), values in [0.0, 1.0].
        - z : torch.Tensor
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
        """Returns the number of learnable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
