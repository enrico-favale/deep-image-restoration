from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from torchvision.models import vgg16

from src.models.autoencoder import ConvAutoencoder
from src.utils.preprocessing import CompressionDataset, get_dataloaders
from src.utils.logger import Logger


class RestorationLoss(nn.Module):
    """
    Combined perceptual and pixel-wise loss for image restoration.

    Combines MSE (pixel accuracy) with a perceptual loss computed
    on intermediate VGG16 features (structural and textural similarity).
    VGG16 weights are frozen and never updated during training.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.1):
        """
        Initialize loss weights and load frozen VGG16 feature extractor.

        Parameters
        ---------------
        - alpha: float, optional
            Weight for the MSE (pixel-wise) term. Default: 1.0.
        - beta: float, optional
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
        Compute the combined restoration loss.

        Parameters
        ---------------
        - pred: torch.Tensor, required
            Restored image predicted by the network.
            Shape: (B, 3, H, W), values in [0.0, 1.0].
        - target: torch.Tensor, required
            Ground truth image.
            Shape: (B, 3, H, W), values in [0.0, 1.0].

        Returns
        ----------
        - loss: torch.Tensor
            Scalar loss value: alpha * MSE + beta * perceptual.
        """
        
        mse = F.mse_loss(pred, target)
        feat_pred = self.vgg(pred)
        feat_target = self.vgg(target)
        perceptual = F.mse_loss(feat_pred, feat_target)

        return self.alpha * mse + self.beta * perceptual


class Trainer:
    """
    Handles dataset loading, training loop, validation, and checkpointing
    for the ConvAutoencoder model.

    Keeps all training logic separate from the model architecture,
    following the single-responsibility principle.
    """

    def __init__(
        self,
        input_dir: Path,
        compressed_dir: Path,
        checkpoints_dir: Path,
        logger: Logger,
        checkpoint_name: str = "best_model",
        image_size: int = 256,
        base_channels: int = 32,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        alpha: float = 1.0,
        beta: float = 0.1,
        splits: tuple = (0.70, 0.15, 0.15),
        seed: int = 42,
    ):
        """
        Initialize the Trainer: device, model, optimizer, scheduler, and loss.

        Does not load data yet — call load_dataset() before train().
        Logs hyperparameters to the provided Logger instance.

        Parameters
        ---------------
        - input_dir: Path, required
            Directory containing the original (ground truth) images.
        - compressed_dir: Path, required
            Directory containing the compressed images (network inputs).
        - checkpoints_dir: Path, required
            Directory where model checkpoints will be saved.
        - checkpoint_name: str
            Name for best model to save.
        - logger: Logger, required
            Logger instance shared with the model for experiment tracking.
        - image_size: int, optional
            Spatial size to which all images are resized. Default: 256.
        - base_channels: int, optional
            Base channel multiplier for ConvAutoencoder. Default: 32.
        - batch_size: int, optional
            Number of samples per training batch. Default: 8.
        - learning_rate: float, optional
            Initial learning rate for Adam optimizer. Default: 1e-3.
        - weight_decay: float, optional
            L2 regularization coefficient for Adam. Default: 1e-5.
        - alpha: float, optional
            Weight for the MSE term in RestorationLoss. Default: 1.0.
        - beta: float, optional
            Weight for the perceptual term in RestorationLoss. Default: 0.1.
        - splits: tuple, optional
            (train, val, test) proportions. Default: (0.70, 0.15, 0.15).
        - seed: int, optional
            Random seed for reproducible dataset splitting. Default: 42.
        """

        self.input_dir = Path(input_dir)
        self.compressed_dir = Path(compressed_dir)
        self.checkpoints_dir = Path(checkpoints_dir)
        self.logger = logger
        self.image_size = image_size
        self.batch_size = batch_size
        self.splits = splits
        self.seed = seed

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_name = f"{checkpoint_name}_{timestamp}.pth"

        # Device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Model — logger passed here to capture architecture
        self.model = ConvAutoencoder(
            base_channels=base_channels,
            logger=logger,
        ).to(self.device)

        # Loss — VGG must be on same device as model
        self.criterion = RestorationLoss(alpha=alpha, beta=beta)
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

        # Log hyperparameters
        logger.log_hyperparameters(
            base_channels=base_channels,
            image_size=image_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            loss_fn=f"MSE (alpha={alpha}) + Perceptual VGG16 (beta={beta})",
            optimizer="Adam",
            # scheduler="ReduceLROnPlateau (mode='min', patience=5, factor=0.5)",
            scheduler="CosineAnnealingLR (T_max=100, eta_min=1e-6)",
            device=str(self.device),
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
        Instantiate the CompressionDataset and build the three DataLoaders.

        Reads image pairs from input_dir and compressed_dir, applies resize
        and normalization, and splits them into train/val/test subsets.
        Logs dataset metadata to the Logger. Must be called before train().
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

        self.logger.log_dataset(
            input_dir=str(self.input_dir),
            compressed_dir=str(self.compressed_dir),
            total_samples=len(self.dataset),
            train_samples=len(self.train_loader.dataset),
            val_samples=len(self.val_loader.dataset),
            test_samples=len(self.test_loader.dataset),
            image_size=self.image_size,
            splits=self.splits,
            seed=self.seed,
        )

        print(f"Device : {self.device}")
        print(f"Parameters : {self.model.count_parameters():,}")
        print(
            f"Train\t| Val\t| Test : \n"
            f"{len(self.train_loader.dataset)}\t| "
            f"{len(self.val_loader.dataset)}\t| "
            f"{len(self.test_loader.dataset)}"
        )

    def _train_epoch(self) -> float:
        """
        Run a single training epoch over the full training DataLoader.

        Parameters
        ---------------
        - None

        Returns
        ----------
        - avg_loss: float
            Average training loss over all batches in the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for compressed, original in self.train_loader:
            compressed = compressed.to(self.device)
            original = original.to(self.device)

            self.optimizer.zero_grad()
            recon, _ = self.model(compressed)
            loss = self.criterion(recon, original)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate_epoch(self) -> float:
        """
        Run a single validation epoch over the full validation DataLoader.

        Parameters
        ---------------
        - None

        Returns
        ----------
        - avg_loss: float
            Average validation loss over all batches in the epoch.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for compressed, original in self.val_loader:
                compressed = compressed.to(self.device)
                original = original.to(self.device)
                recon, _ = self.model(compressed)
                total_loss += self.criterion(recon, original).item()

        return total_loss / len(self.val_loader)


    def train(self, num_epochs: int = 50) -> dict:
        """
        Run the full training loop with validation, checkpointing, and logging.

        Saves the best model checkpoint (lowest val loss) to checkpoints_dir.
        Logs per-epoch metrics to the Logger and calls logger.save() at the end.

        Parameters
        ---------------
        - num_epochs: int, optional
            Number of training epochs. Default: 50.

        Returns
        ----------
        - history: dict
            Dictionary with keys 'train_loss' and 'val_loss',
            each containing a list of per-epoch average losses.
        """
        
        if self.train_loader is None:
            raise RuntimeError("Call load_dataset() before train().")

        self.checkpoints_dir.mkdir(exist_ok=True)
        best_val_loss = float("inf")
        best_epoch = -1

        for epoch in range(1, num_epochs + 1):
            avg_train_loss = self._train_epoch()
            avg_val_loss = self._validate_epoch()
            self.scheduler.step()

            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)

            self.logger.log_epoch(
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                torch.save(
                    self.model.state_dict(),
                    self.checkpoints_dir / self.checkpoint_name,
                )
                self.logger.log_best_results(
                    epoch=best_epoch,
                    val_loss=best_val_loss,
                    checkpoint=str(self.checkpoints_dir / self.checkpoint_name),
                )

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:3d}/{num_epochs}"
                    f"  |  Train Loss: {avg_train_loss:.6f}"
                    f"  |  Val Loss:   {avg_val_loss:.6f}"
                )

        print(
            f"\nTraining complete — Best val loss: {best_val_loss:.6f} (epoch {best_epoch})"
        )
        print(f"Checkpoint saved: {self.checkpoints_dir / self.checkpoint_name}")

        self.logger.save()

        return self.history
