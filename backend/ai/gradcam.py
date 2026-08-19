"""Reusable Grad-CAM implementation for ResNet18 skin classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import cv2  # pyright: ignore[reportMissingImports]
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = ["benign", "malignant", "non-neoplastic"]
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_resnet18(checkpoint_path: str | Path, device: torch.device = DEVICE) -> tuple[torch.nn.Module, list[str], int]:
    """Load a ResNet18 checkpoint saved as a state dict or training dictionary."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict: Any = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a PyTorch state dictionary")

    class_names = list(checkpoint.get("class_names", CLASS_NAMES)) if isinstance(checkpoint, dict) else CLASS_NAMES
    image_size = int(checkpoint.get("image_size", IMAGE_SIZE)) if isinstance(checkpoint, dict) else IMAGE_SIZE

    model = models.resnet18(weights=None)
    if "fc.1.weight" in state_dict:
        classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(model.fc.in_features, len(class_names)),
        )
    else:
        classifier = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.fc = cast(Any, classifier)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, class_names, image_size


class GradCAM:
    """Compute Grad-CAM maps for a convolutional target layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activations)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_inputs, grad_outputs) -> None:
        self.gradients = grad_outputs[0].detach()

    def remove_hooks(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, _exception_type, _exception, _traceback) -> None:
        self.remove_hooks()

    def __call__(self, input_tensor: torch.Tensor, target_class: int | None = None) -> tuple[np.ndarray, int, float]:
        """Return normalized heatmap, predicted class index, and confidence."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = int(logits.argmax(dim=1).item())
        target_class = predicted_class if target_class is None else target_class
        logits[:, target_class].sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not receive activations and gradients")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min()).clamp_min(1e-8)
        cam_uint8 = (cam * 255).clamp(0, 255).byte()
        return cam_uint8.cpu().numpy(), predicted_class, float(probabilities[0, predicted_class].detach().cpu())


def preprocess_image(image: Image.Image, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """Apply the same deterministic preprocessing used during evaluation."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = cast(torch.Tensor, transform(image.convert("RGB")))
    return tensor.unsqueeze(0)


def render_gradcam(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> tuple[Image.Image, Image.Image]:
    """Create RGB heatmap and blended overlay at the original image size."""
    original = image.convert("RGB")
    original_array = np.asarray(original, dtype=np.uint8)
    resized_cam = cv2.resize(heatmap, (original.width, original.height), interpolation=cv2.INTER_LINEAR)
    cam_min = float(resized_cam.min())
    cam_max = float(resized_cam.max())
    if cam_max > cam_min:
        normalized_cam = ((resized_cam - cam_min) * 255.0 / (cam_max - cam_min)).astype(np.uint8)
    else:
        normalized_cam = np.zeros_like(resized_cam, dtype=np.uint8)
    heatmap_bgr = cv2.applyColorMap(normalized_cam, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.addWeighted(original_array, 1 - alpha, heatmap_rgb, alpha, 0)
    heatmap_image = Image.fromarray(heatmap_rgb)
    overlay = Image.fromarray(overlay_rgb)
    return heatmap_image, overlay


def generate_gradcam(image_path: str | Path, checkpoint_path: str | Path | None = None, output_dir: str | Path | None = None) -> dict:
    """Generate and save Grad-CAM assets for one image."""
    checkpoint_path = checkpoint_path or PROJECT_ROOT / "models" / "resnet18_best.pth"
    output_dir = Path(output_dir or PROJECT_ROOT / "reports" / "gradcam_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    model, class_names, image_size = build_resnet18(checkpoint_path)
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess_image(image, image_size).to(DEVICE)
    target_layer = cast(Any, model).layer4[-1]
    with GradCAM(model, target_layer) as gradcam:
        heatmap, predicted_index, confidence = gradcam(input_tensor)
    heatmap_image, overlay = render_gradcam(image, heatmap)
    stem = image_path.stem
    paths = {
        "original": output_dir / f"{stem}_original.jpg",
        "heatmap": output_dir / f"{stem}_heatmap.jpg",
        "overlay": output_dir / f"{stem}_overlay.jpg",
    }
    image.save(paths["original"], quality=95)
    heatmap_image.save(paths["heatmap"], quality=95)
    overlay.save(paths["overlay"], quality=95)
    return {
        "prediction": class_names[predicted_index],
        "confidence": round(confidence * 100, 2),
        "heatmap": f"/reports/{paths['heatmap'].name}",
        "overlay": f"/reports/{paths['overlay'].name}",
    }
