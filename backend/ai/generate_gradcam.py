"""Generate Grad-CAM visualizations for correct and incorrect test predictions.

Example:
    .venv/Scripts/python.exe -m backend.ai.generate_gradcam --num-correct 20 --num-incorrect 20
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from PIL import Image

from backend.ai.gradcam import DEVICE, GradCAM, build_resnet18, preprocess_image, render_gradcam


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("dermaguard.gradcam")


def save_visualization(model, class_names, image_path: Path, true_label: str, image_size: int, output_dir: Path) -> dict:
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess_image(image, image_size).to(DEVICE)
    with GradCAM(model, model.layer4[-1]) as gradcam:
        heatmap, predicted_index, confidence = gradcam(input_tensor)
    heatmap_image, overlay = render_gradcam(image, heatmap)
    predicted_label = class_names[predicted_index]
    sample_dir = output_dir / ("correct" if predicted_label == true_label else "incorrect")
    sample_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    paths = {
        "original": sample_dir / f"{stem}_original.jpg",
        "heatmap": sample_dir / f"{stem}_heatmap.jpg",
        "overlay": sample_dir / f"{stem}_overlay.jpg",
    }
    image.save(paths["original"], quality=95)
    heatmap_image.save(paths["heatmap"], quality=95)
    overlay.save(paths["overlay"], quality=95)
    return {
        "image": str(image_path),
        "true_label": true_label,
        "predicted_label": predicted_label,
        "correct": predicted_label == true_label,
        "confidence": round(confidence * 100, 2),
        "original": str(paths["original"]),
        "heatmap": str(paths["heatmap"]),
        "overlay": str(paths["overlay"]),
    }


def generate(args: argparse.Namespace) -> list[dict]:
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    model, class_names, image_size = build_resnet18(checkpoint_path, DEVICE)
    test_frame = pd.read_csv(args.test_csv)
    shuffled_frame = test_frame.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    correct_count, incorrect_count, manifest = 0, 0, []

    for row in shuffled_frame.itertuples(index=False):
        if correct_count >= args.num_correct and incorrect_count >= args.num_incorrect:
            break
        true_label = str(row.three_partition_label)
        image_path = PROJECT_ROOT / str(row.image_path)
        if not image_path.is_file():
            LOGGER.warning("Skipping missing image: %s", image_path)
            continue
        result = save_visualization(model, class_names, image_path, true_label, image_size, output_dir)
        if result["correct"]:
            if correct_count >= args.num_correct:
                for key in ("original", "heatmap", "overlay"):
                    Path(result[key]).unlink(missing_ok=True)
                continue
            correct_count += 1
        else:
            if incorrect_count >= args.num_incorrect:
                for key in ("original", "heatmap", "overlay"):
                    Path(result[key]).unlink(missing_ok=True)
                continue
            incorrect_count += 1
        manifest.append(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Saved %d correct and %d incorrect visualizations to %s", correct_count, incorrect_count, output_dir)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "models" / "resnet18_baseline_best.pth"))
    parser.add_argument("--test-csv", default=str(PROJECT_ROOT / "datasets" / "fitzpatrick17k" / "test.csv"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports" / "gradcam_examples"))
    parser.add_argument("--num-correct", type=int, default=20)
    parser.add_argument("--num-incorrect", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate(parse_args())
