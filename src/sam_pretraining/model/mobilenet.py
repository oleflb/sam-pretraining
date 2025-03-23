from torch import nn
import torch
import timm


class MobileNetSam(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "mobilenetv2_035", pretrained=False, features_only=True, out_indices=(2,)
        )
        self.adaptor = nn.Conv2d(16, 256, (13, 17))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        return self.adaptor(features[0])
