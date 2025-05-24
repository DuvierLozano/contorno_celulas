import segmentation_models_pytorch as smp
import torch

def get_model(encoder='resnet101', weights='imagenet'):
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=3,
        classes=1
    )