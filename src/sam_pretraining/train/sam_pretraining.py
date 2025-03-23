import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm, trange
from torchmetrics.aggregation import MeanMetric
import wandb
from dataclasses import dataclass, asdict

from sam_pretraining.utils import get_default_device


@dataclass
class SamPretrainerHyperparameters:
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 10


class SamPretrainer:
    def __init__(
        self,
        *,
        student: nn.Module,
        train_dataset: Dataset,
        validation_dataset: None | Dataset = None,
        hyperparameters: SamPretrainerHyperparameters = None,
        device: torch.device = get_default_device(),
    ):
        self.student = student.to(device)
        self.train_dataset = train_dataset
        self.hyperparameters = hyperparameters or SamPretrainerHyperparameters()

        self.optimizer = torch.optim.AdamW(
            student.parameters(), lr=self.hyperparameters.learning_rate
        )
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.hyperparameters.batch_size, shuffle=True
        )
        self.val_dataloader = (
            DataLoader(validation_dataset, batch_size=self.hyperparameters.batch_size)
            if validation_dataset is not None
            else None
        )
        self.device = device
        wandb.config.update(asdict(self.hyperparameters))

    def train_step(
        self, images: torch.Tensor, embeddings: torch.Tensor
    ) -> torch.Tensor:
        self.optimizer.zero_grad()
        student_embeddings = self.student(images)
        loss = F.mse_loss(student_embeddings, embeddings)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    @torch.inference_mode()
    def val_step(self, images: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        student_embeddings = self.student(images)
        loss = F.mse_loss(student_embeddings, embeddings)
        return loss

    def train_epoch(self) -> torch.Tensor:
        loss_tracker = MeanMetric().to(self.device)
        self.student.train()
        for images, embeddings in (pbar := tqdm(self.train_dataloader)):
            loss = self.train_step(
                images.to(self.device),
                embeddings.to(self.device),
            )
            pbar.set_description_str(f"Training Loss: {loss:.4f}")
            loss_tracker.update(loss)
        return loss_tracker.compute()

    def val_epoch(self) -> torch.Tensor:
        self.student.eval()
        if self.val_dataloader is None:
            return torch.tensor(torch.nan)

        loss_tracker = MeanMetric().to(self.device)
        for images, embeddings in tqdm(self.val_dataloader, desc="Validation"):
            loss = self.val_step(
                images.to(self.device),
                embeddings.to(self.device),
            )
            loss_tracker.update(loss)
        return loss_tracker.compute()

    def train(self):
        for epoch in (pbar := trange(self.hyperparameters.epochs)):
            train_loss = self.train_epoch()
            val_loss = self.val_epoch()
            pbar.set_description_str(
                f"{epoch + 1}/{self.hyperparameters.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
            )
            wandb.log({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
