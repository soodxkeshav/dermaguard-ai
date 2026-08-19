"""Run top-k predictions for a decoded skin lesion image or file path."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from model_loader import CLASS_NAMES, load_model
from utils import preprocess_image


def predict_image(image: Image.Image, model: torch.nn.Module, device: torch.device, top_k: int = 3):
    """Return labels and confidence percentages ordered from highest to lowest."""
    tensor = preprocess_image(image).to(device)
    if device.type == "cuda":
        tensor = tensor.to(memory_format=torch.channels_last)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    count = min(top_k, len(CLASS_NAMES))
    scores, indices = torch.topk(probabilities, count)
    return [
        {"label": CLASS_NAMES[index.item()], "confidence": float(score.item()) * 100}
        for score, index in zip(scores, indices)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a skin lesion image")
    parser.add_argument("image", type=Path, help="Path to a JPG, JPEG, or PNG image")
    parser.add_argument("--model", type=Path, default=None, help="Optional checkpoint path")
    parser.add_argument("--top-k", type=int, default=3, choices=range(1, len(CLASS_NAMES) + 1))
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    try:
        with Image.open(args.image) as image:
            decoded_image = image.convert("RGB")
        model, device = load_model(args.model) if args.model else load_model()
        result = predict_image(decoded_image, model, device, args.top_k)
    except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
