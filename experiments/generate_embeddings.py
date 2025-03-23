from pathlib import Path
from PIL.ImageFile import ImageFile
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sam_pretraining.sam import EmbeddingGenerator
from sam_pretraining.dataset import ImageDataset
from sam_pretraining.utils import crop_embedding


def collate_fn(
    batch: list[tuple[ImageFile, Path]],
) -> tuple[list[ImageFile], list[Path]]:
    images, paths = zip(*batch)
    return list(images), list(paths)


def embedding_path_from_image_path(image_path: Path) -> Path:
    return Path("embeddings") / image_path.with_suffix(".pt").name


def main():
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2

    model = EmbeddingGenerator(device=device, dtype=dtype)
    dataset = ImageDataset(Path("images"))
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    for images, paths in tqdm(dataloader):
        embedding_paths = [embedding_path_from_image_path(path) for path in paths]
        if all(embedding_path.exists() for embedding_path in embedding_paths):
            continue

        embeddings = model.generate_embeddings(images)
        for path, image, embedding in zip(embedding_paths, images, embeddings):
            cropped = crop_embedding(embedding, image.size)
            torch.save(cropped, path)


if __name__ == "__main__":
    main()
