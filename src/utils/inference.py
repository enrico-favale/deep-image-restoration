from PIL import Image
from torchvision import transforms
import torch

from src.models.autoencoder import ConvAutoencoder


def restore_image(
    compressed_path: str,
    checkpoint_path: str,
    base_channels: int = 32,
    device: str = "cpu",
) -> Image.Image:

    model = ConvAutoencoder(base_channels=base_channels)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    img = Image.open(compressed_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        restored, _ = model(x)

    out = restored.squeeze(0).clamp(0, 1)
    
    return transforms.ToPILImage()(out)
