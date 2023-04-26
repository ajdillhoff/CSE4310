import torch.nn as nn

from torchvision.models import vision_transformer


class ViTHPE(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # Load ViT backbone pretrained on ImageNet1K
        self.vit = vision_transformer.vit_b_16(weights=vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1)

        # TODO: Replace the head with a new one

    def forward(self, x):
        x = self.vit(x)
        return x