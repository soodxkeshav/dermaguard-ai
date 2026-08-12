import torch
import torch.nn as nn
from torchvision import models


# ============================================================
# DERMA GUARD AI — BASELINE MODEL
# ResNet18 Image Classification
# ============================================================


NUM_CLASSES = 3


def create_model(num_classes=NUM_CLASSES):
    """
    Create a pretrained ResNet18 model
    and replace the final classifier.
    """

    # Load pretrained ResNet18
    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(weights=weights)

    # Get number of inputs to final layer
    input_features = model.fc.in_features

    # Replace ImageNet classifier with our 3-class classifier
    model.fc = nn.Linear(
        input_features,
        num_classes
    )

    return model


def get_device():
    """
    Automatically select GPU if available.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")

        print("Device: CUDA")
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    else:
        device = torch.device("cpu")

        print("Device: CPU")

    return device