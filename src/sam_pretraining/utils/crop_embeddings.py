import torch


def crop_embedding(
    embedding: torch.FloatTensor, image_size: tuple[int, int]
) -> torch.FloatTensor:
    assert embedding.dim() == 3, "Embedding must have 3 dimensions"
    image_aspect_ratio = image_size[0] / image_size[1]
    embedding_aspect_ratio = embedding.size(1) / embedding.size(2)

    if image_aspect_ratio > embedding_aspect_ratio:
        new_width = embedding.size(1)
        new_height = int(embedding.size(1) / image_aspect_ratio)
    else:
        new_width = int(embedding.size(2) * image_aspect_ratio)
        new_height = embedding.size(2)

    return embedding[:, :new_height, :new_width]
