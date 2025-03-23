from sam_pretraining.utils import get_default_device
from .setup import download_sam_model, download_sam_processor

from PIL import Image
import torch


class EmbeddingGenerator:
    def __init__(
        self, device: torch.device = get_default_device(), dtype=torch.float32
    ):
        self.device = device
        self.dtype = dtype
        self.processor = download_sam_processor()
        model = download_sam_model().to(device)
        self.vision_encoder = model.vision_encoder.to(dtype=dtype, device=device)

    @torch.no_grad()
    @torch.compile()
    def __forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_output = self.vision_encoder(
            pixel_values,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        image_embeddings = vision_output[0]
        return image_embeddings

    @torch.inference_mode()
    def generate_embeddings(self, image: Image) -> torch.Tensor:
        inputs = self.processor(image, return_tensors="pt")
        return self.__forward(
            inputs["pixel_values"].to(dtype=self.dtype, device=self.device)
        )
