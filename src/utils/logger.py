from pathlib import Path
from datetime import datetime


class Logger:
    """
    Experiment logger for deep learning training runs.

    Collects architecture, hyperparameter, dataset, and training
    information from different components and writes a formatted
    log file upon completion of the experiment.

    Parameters
    ---------------
    - dir: Path, optional
        Directory where log files will be saved.
    - description: str, optional
        Short label appended to the log filename to identify the experiment.
    """

    def __init__(
        self,
        dir: Path = Path("../../logs"),
        description: str = "",
    ) -> None:
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)

        self.description = description

        self.architecture: str = ""
        self.hyperparameters: dict = {}
        self.dataset_info: dict = {}
        self.training_info: list[str] = []
        self.best_results_info: dict = {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = f"_{description}" if description else ""
        self.log_filename = self.dir / f"{timestamp}{slug}.log"

    def log_architecture(self, model_name: str, model_summary: str) -> None:
        """
        Store the model architecture description.

        Should be called inside the model class after layer definition,
        typically passing str(self) or a torchinfo summary.

        Parameters
        ---------------
        - model_name: str, required
            Human-readable name of the architecture (e.g. 'RestorationAutoencoder').
        - model_summary: str, required
            Full string representation of the model layers and parameters.
        """

        self.architecture = f"Model: {model_name}\n{model_summary}"

    def log_hyperparameters(self, **kwargs) -> None:
        """
        Store training hyperparameters as key-value pairs.

        Should be called in the Trainer before the training loop starts.

        Parameters
        ---------------
        - **kwargs: any, required
            Arbitrary hyperparameter names and values
            (e.g. learning_rate=1e-3, epochs=100, batch_size=32).
        """

        self.hyperparameters = kwargs

    def log_dataset(self, **kwargs) -> None:
        """
        Store dataset metadata as key-value pairs.

        Should be called in the Trainer or DataLoader wrapper after
        dataset initialization.

        Parameters
        ---------------
        - **kwargs: any, required
            Arbitrary dataset properties
            (e.g. train_samples=800, image_size=256).
        """

        self.dataset_info = kwargs

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
    ) -> None:
        """
        Append a per-epoch metrics entry to the training log.

        Prints the entry to stdout in real time and stores it
        internally for later file serialization.

        Parameters
        ---------------
        - epoch: int, required
            Current epoch index (0-based).
        - train_loss: float, required
            Average training loss over the epoch.
        - val_loss: float, required
            Average validation loss over the epoch.
        """

        entry = (
            f"Epoch {epoch:04d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )
        self.training_info.append(entry)

    def log_best_results(self, **kwargs) -> None:
        """
        Store the best checkpoint metrics, overwriting any previous entry.

        Should be called whenever a new best model is saved during training.

        Parameters
        ---------------
        - **kwargs: any, required
            Metrics and metadata at the best checkpoint
            (e.g. epoch=47, psnr=34.21, ssim=0.93, checkpoint='best_model.pth').
        """

        self.best_results_info = kwargs

    def save(self) -> Path:
        """
        Write all collected information to a formatted .log file.

        Serializes all stored fields into clearly delimited sections.
        Should be called once at the end of the training run.

        Returns
        ----------
        - log_filename: Path
            Absolute path to the written log file.
        """

        lines = []

        def section(title: str, content: str) -> None:
            lines.append(f"\n{'='*60}")
            lines.append(f"  {title.upper()}")
            lines.append(f"{'='*60}")
            lines.append(content)

        lines.append(f"EXPERIMENT LOG — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.description:
            lines.append(f"Description: {self.description}")

        if self.architecture:
            section("Architecture", self.architecture)

        if self.hyperparameters:
            content = "\n".join(f"  {k}: {v}" for k, v in self.hyperparameters.items())
            section("Hyperparameters", content)

        if self.dataset_info:
            content = "\n".join(f"  {k}: {v}" for k, v in self.dataset_info.items())
            section("Dataset", content)

        if self.training_info:
            section("Training Log", "\n".join(f"  {e}" for e in self.training_info))

        if self.best_results_info:
            content = "\n".join(
                f"  {k}: {v}" for k, v in self.best_results_info.items()
            )
            section("Best Results", content)

        self.log_filename.write_text("\n".join(lines), encoding="utf-8")
        print(f"[Logger] Log saved to: {self.log_filename}")

        return self.log_filename
