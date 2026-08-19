"""Model loading and image classification for the DermaGuard API."""

from pathlib import Path
from typing import Any, TypedDict

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


CLASSES = [
	"benign",
	"malignant",
	"non-neoplastic",
]

IMAGE_SIZE = 224
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = _PROJECT_ROOT / "models" / "resnet18_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PredictionResult(TypedDict):
	prediction: str
	confidence: float

_transform = transforms.Compose([
	transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
	transforms.ToTensor(),
	transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225],
	),
])


def _build_model() -> torch.nn.Module:
	if not MODEL_PATH.is_file():
		raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

	model = models.resnet18(weights=None)
	model.fc = torch.nn.Sequential(
		torch.nn.Dropout(0.4),
		torch.nn.Linear(model.fc.in_features, len(CLASSES)),
	)
	checkpoint: Any = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

	if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
		checkpoint = checkpoint["model_state_dict"]
	if not isinstance(checkpoint, dict):
		raise ValueError("Model checkpoint must contain a state dictionary")

	model.load_state_dict(checkpoint)
	model.to(DEVICE)
	model.eval()
	return model


model = _build_model()


def predict_image(image_path: str | Path) -> PredictionResult:
	"""Classify an image from disk and return its class and percentage confidence."""
	try:
		with Image.open(image_path) as image:
			tensor = _transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
	except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
		raise ValueError("The uploaded file is not a valid image") from exc

	with torch.inference_mode():
		probabilities = torch.softmax(model(tensor), dim=1)[0]

	confidence, class_index = torch.max(probabilities, dim=0)
	return {
		"prediction": CLASSES[class_index.item()],
		"confidence": round(float(confidence.item()) * 100, 2),
	}
