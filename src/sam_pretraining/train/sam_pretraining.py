from typing import Literal
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm, trange
from torchmetrics.aggregation import MeanMetric
import wandb
from dataclasses import dataclass, asdict
from accelerate import Accelerator
from accelerate.utils import PrecisionType

from sam_pretraining.utils import Checkpointer


@dataclass
class SamPretrainerHyperparameters:
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    training_precision: PrecisionType | Literal["no", "fp8", "fp16", "bf16"] = (
        PrecisionType.NO
    )


class SamPretrainer:
    student: nn.Module
    train_dataloader: DataLoader
    val_dataloader: None | DataLoader
    hyperparameters: SamPretrainerHyperparameters
    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        *,
        student: nn.Module,
        train_dataset: Dataset,
        validation_dataset: None | Dataset = None,
        hyperparameters: SamPretrainerHyperparameters = None,
    ):
        self.accelerator = Accelerator(
            mixed_precision=hyperparameters.training_precision,
            log_with="wandb",
        )
        self.checkpointer = Checkpointer()
        self.train_dataset = train_dataset
        self.hyperparameters = hyperparameters or SamPretrainerHyperparameters()
        wandb.config.update(asdict(self.hyperparameters))

        optimizer = torch.optim.AdamW(
            student.parameters(), lr=self.hyperparameters.learning_rate
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.hyperparameters.batch_size, shuffle=True
        )
        val_dataloader = (
            DataLoader(validation_dataset, batch_size=self.hyperparameters.batch_size)
            if validation_dataset is not None
            else None
        )

        (self.student, self.optimizer, self.train_dataloader, self.val_dataloader) = (
            self.accelerator.prepare(
                student, optimizer, train_dataloader, val_dataloader
            )
        )

    def train_step(
        self, images: torch.Tensor, embeddings: torch.Tensor
    ) -> torch.Tensor:
        self.optimizer.zero_grad()
        student_embeddings = self.student(images)
        loss = F.mse_loss(student_embeddings, embeddings)
        self.accelerator.backward(loss)
        self.optimizer.step()
        return loss.detach()

    @torch.inference_mode()
    def val_step(self, images: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        student_embeddings = self.student(images)
        loss = F.mse_loss(student_embeddings, embeddings)
        return loss

    def train_epoch(self) -> torch.Tensor:
        loss_tracker = MeanMetric().to(self.accelerator.device)
        self.student.train()
        for images, embeddings in (pbar := tqdm(self.train_dataloader)):
            loss = self.train_step(images, embeddings)
            pbar.set_description_str(f"Training Loss: {loss:.4f}")
            loss_tracker.update(loss)
        return loss_tracker.compute()

    def val_epoch(self, epoch: int) -> torch.Tensor:
        self.student.eval()
        if self.val_dataloader is None:
            return torch.tensor(torch.nan)

        loss_tracker = MeanMetric().to(self.accelerator.device)
        for images, embeddings in tqdm(self.val_dataloader, desc="Validation"):
            loss = self.val_step(images, embeddings)
            loss_tracker.update(loss)

        final_loss = loss_tracker.compute()
        self.checkpointer.checkpoint_if_improved(
            final_loss.item(), epoch=epoch, model=self.student, optimizer=self.optimizer
        )
        return final_loss

    def train(self):
        for epoch in (pbar := trange(self.hyperparameters.epochs)):
            train_loss = self.train_epoch()
            val_loss = self.val_epoch(epoch)
            pbar.set_description_str(
                f"{epoch + 1}/{self.hyperparameters.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
            )
            wandb.log({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
