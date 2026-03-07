import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from torchvision.models import vgg16

from src.models.autoencoder import ConvAutoencoder
from src.utils.preprocessing import CompressionDataset, get_dataloaders


class RestorationLoss(nn.Module):
    """
    Combined perceptual and pixel-wise loss for image restoration.

    Combines MSE (pixel accuracy) with a perceptual loss computed
    on intermediate VGG16 features (structural and textural similarity).
    VGG16 weights are frozen and never updated during training.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.1):
        """
        Parameters
        ----------
        alpha : float, optional
            Weight for the MSE (pixel-wise) term. Default: 1.0.
        beta : float, optional
            Weight for the perceptual (VGG feature) term. Default: 0.1.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        vgg = vgg16(weights="DEFAULT").features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the combined restoration loss.

        Parameters
        ----------
        pred : torch.Tensor, required
            Restored image predicted by the network, shape (B, 3, H, W).
        target : torch.Tensor, required
            Ground truth image, shape (B, 3, H, W).

        Returns
        -------
        loss : torch.Tensor
            Scalar loss value: alpha * MSE + beta * perceptual.
        """
        mse = F.mse_loss(pred, target)

        feat_pred = self.vgg(pred)
        feat_target = self.vgg(target)
        perceptual = F.mse_loss(feat_pred, feat_target)

        return self.alpha * mse + self.beta * perceptual


class Trainer:
    """
    Handles dataset loading, training loop, validation and checkpointing
    for the ConvAutoencoder model.

    Keeps all training logic separate from the model architecture,
    following the single-responsibility principle.
    """

    def __init__(
        self,
        input_dir: Path,
        compressed_dir: Path,
        checkpoints_dir: Path,
        image_size: int = 256,
        base_channels: int = 32,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        splits: tuple = (0.70, 0.15, 0.15),
        seed: int = 42,
    ):
        """
        Initializes the Trainer and sets up device, model, optimizer and
        loss function. Does not load data yet — call load_dataset() first.

        Parameters
        ----------
        input_dir : Path, required
            Directory containing the original (ground truth) images.
        compressed_dir : Path, required
            Directory containing the compressed images (network inputs).
        checkpoints_dir : Path, required
            Directory where model checkpoints will be saved.
        image_size : int, optional
            Spatial size to which all images are resized. Default: 256.
        base_channels : int, optional
            Base channel multiplier for ConvAutoencoder. Default: 32.
        batch_size : int, optional
            Number of samples per training batch. Default: 8.
        learning_rate : float, optional
            Initial learning rate for Adam optimizer. Default: 1e-3.
        weight_decay : float, optional
            L2 regularization coefficient for Adam. Default: 1e-5.
        splits : tuple, optional
            (train, val, test) proportions. Default: (0.70, 0.15, 0.15).
        seed : int, optional
            Random seed for reproducible dataset splitting. Default: 42.
        """
        self.input_dir = Path(input_dir)
        self.compressed_dir = Path(compressed_dir)
        self.checkpoints_dir = Path(checkpoints_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.splits = splits
        self.seed = seed
        self.checkpoint_name = "best_model_autoencoder.pth"

        # Device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Model
        self.model = ConvAutoencoder(
            base_channels=base_channels,
        ).to(self.device)

        # Loss — VGG must be on same device as model
        self.criterion = RestorationLoss(alpha=1.0, beta=0.1)
        self.criterion.vgg = self.criterion.vgg.to(self.device)

        # Optimizer & scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode="min", patience=5, factor=0.5
        # )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        # Populated by load_dataset()
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Populated by train()
        self.history = {"train_loss": [], "val_loss": []}

    def load_dataset(self) -> None:
        """
        Instantiates the CompressionDataset and builds the three DataLoaders.

        Reads image pairs from input_dir and compressed_dir, applies resize
        and normalization, and splits them into train/val/test subsets.
        Must be called before train().

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.dataset = CompressionDataset(
            input_dir=self.input_dir,
            compressed_dir=self.compressed_dir,
            image_size=self.image_size,
        )

        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            self.dataset,
            splits=self.splits,
            batch_size=self.batch_size,
            seed=self.seed,
        )

        print(f"Device          : {self.device}")
        print(f"Parameters      : {self.model.count_parameters():,}")
        print(
            f"Train / Val / Test : "
            f"{len(self.train_loader.dataset)} / "
            f"{len(self.val_loader.dataset)} / "
            f"{len(self.test_loader.dataset)}"
        )

    def train(self, num_epochs: int = 50) -> dict:
        """
        Runs the full training loop with validation and checkpointing.

        Saves the best model (lowest val loss) to checkpoints_dir
        as best_model_autoencoder.pth.

        Parameters
        ----------
        num_epochs : int, optional
            Number of training epochs. Default: 50.

        Returns
        -------
        history : dict
            Dictionary with keys "train_loss" and "val_loss",
            each containing a list of per-epoch average losses.

        Raises
        ------
        RuntimeError
            If load_dataset() has not been called before train().
        """
        if self.train_loader is None:
            raise RuntimeError("Call load_dataset() before train().")

        self.checkpoints_dir.mkdir(exist_ok=True)
        best_val_loss = float("inf")

        for epoch in range(1, num_epochs + 1):

            # Train phase
            self.model.train()
            train_loss = 0.0

            for compressed, original in self.train_loader:
                compressed = compressed.to(self.device)
                original = original.to(self.device)

                self.optimizer.zero_grad()
                recon, _ = self.model(compressed)
                loss = self.criterion(recon, original)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(self.train_loader)

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for compressed, original in self.val_loader:
                    compressed = compressed.to(self.device)
                    original = original.to(self.device)
                    recon, _ = self.model(compressed)
                    val_loss += self.criterion(recon, original).item()

            avg_val_loss = val_loss / len(self.val_loader)
            self.scheduler.step(avg_val_loss)

            # History & checkpoint
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    self.model.state_dict(), self.checkpoints_dir / self.checkpoint_name
                )

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:3d}/{num_epochs}"
                    f"  |  Train Loss: {avg_train_loss:.6f}"
                    f"  |  Val Loss:   {avg_val_loss:.6f}"
                )

        print(f"\n✓ Training complete. Best val loss: {best_val_loss:.6f}")
        print(f"  Checkpoint saved: {self.checkpoints_dir / self.checkpoint_name}")
        return self.history
