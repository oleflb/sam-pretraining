import torch
from torch import nn
from pathlib import Path


class Checkpointer:
    def __init__(self):
        self.checkpoint_dir = Path("checkpoints")
        self.current_metric = float("inf")

    def checkpoint_if_improved(
        self,
        metric: float,
        *,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ):
        if metric < self.current_metric:
            self.current_metric = metric
            self.save_checkpoint(
                epoch=epoch, optimizer=optimizer, model=model, scheduler=scheduler
            )

    def save_checkpoint(
        self,
        *,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    ):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
            if scheduler is not None
            else None,
        }

        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        # Save the checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
