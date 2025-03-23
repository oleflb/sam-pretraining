from torch.utils.data import random_split
import wandb

from sam_pretraining.dataset import SamEmbeddingDataset
from sam_pretraining.train import SamPretrainer, SamPretrainerHyperparameters
from sam_pretraining.model import MobileNetSam


def main():
    model = MobileNetSam()
    dataset = SamEmbeddingDataset(
        image_root="images",
        embedding_root="embeddings",
    )
    train_set, validation_set = random_split(dataset, [0.8, 0.2])

    wandb.init(project="sam-pretraining", mode="online")

    pretrainer = SamPretrainer(
        student=model,
        train_dataset=train_set,
        validation_dataset=validation_set,
        hyperparameters=SamPretrainerHyperparameters(
            epochs=500
        ),
    )
    pretrainer.train()


if __name__ == "__main__":
    main()
