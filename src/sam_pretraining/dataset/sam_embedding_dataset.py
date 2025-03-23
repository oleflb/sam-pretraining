from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import to_dtype_image
from torchvision.io.image import decode_image


class SamEmbeddingDataset(Dataset):
    def __init__(self, *, image_root: Path | str, embedding_root: Path | str):
        image_names = set(path.stem for path in Path(image_root).glob("*.png"))
        embedding_names = set(path.stem for path in Path(embedding_root).glob("*.pt"))

        intersection = image_names.intersection(embedding_names)
        print(f"Found {len(intersection)} images with embeddings")

        self.image_paths = [Path(image_root) / f"{name}.png" for name in intersection]
        self.embedding_paths = [
            Path(embedding_root) / f"{name}.pt" for name in intersection
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = to_dtype_image(
            decode_image(str(self.image_paths[index])),
            scale=True,
        )
        embedding = torch.load(self.embedding_paths[index])
        return image, embedding
