from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from PIL.ImageFile import ImageFile


class ImageDataset(Dataset):
    def __init__(self, root: Path, extension: str = ".png"):
        self.paths = list(root.glob(f"*{extension}"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int) -> ImageFile:
        path = self.paths[index]
        return Image.open(path), path
