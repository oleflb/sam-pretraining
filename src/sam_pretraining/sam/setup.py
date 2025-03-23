from transformers import SamProcessor, SamModel

VARIANT = "facebook/sam-vit-base"


def download_sam_processor() -> SamProcessor:
    return SamProcessor.from_pretrained(VARIANT)


def download_sam_model() -> SamModel:
    return SamModel.from_pretrained(VARIANT)
