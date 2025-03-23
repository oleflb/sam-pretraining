from tqdm import trange
from sam_pretraining.dataset import SamEmbeddingDataset

def main():
    dataset = SamEmbeddingDataset(
        image_root="images",
        embedding_root="embeddings",
    )
    length = len(dataset)
    try:
        for i in trange(length):
            _ = dataset[i]
    except Exception as e:
        offending_path = dataset.image_paths[i]
        print(f"Error at index {i} ({offending_path}): {e}")

if __name__ == "__main__":
    main()