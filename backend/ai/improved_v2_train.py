"""Train and evaluate ResNet18 Improved V2 with a class-balanced sampler.

Run from the repository root with:
    .venv/Scripts/python.exe -m backend.ai.improved_v2_train

The script uses the existing Fitzpatrick17k CSV splits. It balances class
exposure in training with WeightedRandomSampler, while validation and test
remain untouched. Fitzpatrick group metrics are descriptive fairness
measurements, not a substitute for clinical validation.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from backend.ai.metrics import (
    classification_metrics,
    fairness_metrics,
    save_comparison_table,
    save_confusion_matrix,
    save_fairness_plot,
    save_training_curves,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "datasets" / "fitzpatrick17k"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "improved_v2"
CLASS_NAMES = ["benign", "malignant", "non-neoplastic"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_SIZE = 224
DEFAULT_SEED = 42
LOGGER = logging.getLogger("dermaguard.improved_v2")


class FitzpatrickDataset(Dataset):
    """CSV-backed dataset returning image tensor, target, and row index."""

    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        image_path = PROJECT_ROOT / str(row["image_path"])
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        label = CLASS_TO_INDEX[str(row["three_partition_label"])]
        return tensor, label


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_sampler(frame: pd.DataFrame, seed: int) -> WeightedRandomSampler:
    """Create per-example inverse-class-frequency sampling weights."""
    labels = frame["three_partition_label"].map(CLASS_TO_INDEX).to_numpy()
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Every class needs training examples; counts={counts.tolist()}")
    sample_weights = torch.as_tensor(1.0 / counts[labels], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def create_model(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def run_epoch(model, loader, criterion, optimizer, device, training: bool) -> dict:
    model.train(training)
    running_loss = 0.0
    labels, predictions = [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            if training:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * images.size(0)
            labels.extend(targets.detach().cpu().tolist())
            predictions.extend(outputs.argmax(1).detach().cpu().tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=range(len(CLASS_NAMES)), average="macro", zero_division=0
    )
    return {
        "loss": running_loss / len(loader.dataset),
        "accuracy": float(np.mean(np.asarray(labels) == np.asarray(predictions))),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "labels": labels,
        "predictions": predictions,
    }


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_frame = pd.read_csv(DATASET_DIR / "train.csv")
    val_frame = pd.read_csv(DATASET_DIR / "validation.csv")
    test_frame = pd.read_csv(DATASET_DIR / "test.csv")
    for frame in (train_frame, val_frame, test_frame):
        unknown = set(frame["three_partition_label"]) - set(CLASS_NAMES)
        if unknown:
            raise ValueError(f"Unexpected class labels: {sorted(unknown)}")

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = FitzpatrickDataset(train_frame, train_transform)
    val_dataset = FitzpatrickDataset(val_frame, eval_transform)
    test_dataset = FitzpatrickDataset(test_frame, eval_transform)
    sampler = create_sampler(train_frame, args.seed)
    loader_args = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    model = create_model(not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    checkpoint_path = MODEL_DIR / "resnet18_improved_v2_best.pth"
    best_f1, best_epoch, history = -1.0, 0, []

    LOGGER.info("Training V2 on %s (%s)", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    LOGGER.info("Class counts before sampling: %s", train_frame["three_partition_label"].value_counts().to_dict())
    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_result = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step(val_result["macro_f1"])
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value for key, value in train_result.items() if key not in {"labels", "predictions"}})
        row.update({f"val_{key}": value for key, value in val_result.items() if key not in {"labels", "predictions"}})
        history.append(row)
        LOGGER.info("Epoch %d/%d train_f1=%.4f val_f1=%.4f", epoch, args.epochs, train_result["macro_f1"], val_result["macro_f1"])
        if val_result["macro_f1"] > best_f1:
            best_f1, best_epoch = val_result["macro_f1"], epoch
            torch.save({"model_state_dict": model.state_dict(), "class_names": CLASS_NAMES, "image_size": IMAGE_SIZE, "best_epoch": epoch, "best_val_macro_f1": best_f1, "seed": args.seed}, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_result = run_epoch(model, test_loader, criterion, optimizer, device, False)
    overall = classification_metrics(test_result["labels"], test_result["predictions"], CLASS_NAMES)
    fairness = fairness_metrics(test_frame, test_result["labels"], test_result["predictions"])

    result = {
        "model": "ResNet18 Improved V2",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "dataset": {"training_samples": len(train_frame), "validation_samples": len(val_frame), "test_samples": len(test_frame)},
        "configuration": vars(args),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "test": {"loss": test_result["loss"], **{key: overall[key] for key in ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1")}},
        "fairness": fairness,
        "confusion_matrix": overall["confusion_matrix"],
        "history": history,
        "model_path": str(checkpoint_path),
    }
    results_path = REPORT_DIR / "improved_v2_results.json"
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(fairness["groups"]).T.to_csv(REPORT_DIR / "fairness_by_fitzpatrick_group.csv")
    save_confusion_matrix(overall["confusion_matrix"], CLASS_NAMES, REPORT_DIR / "confusion_matrix.png", "Improved V2 confusion matrix")
    save_training_curves(history, REPORT_DIR / "training_curves.png")
    save_fairness_plot(fairness, REPORT_DIR / "fairness_by_skin_tone.png")
    save_comparison_table({"Baseline": REPORT_DIR.parent / "baseline_results.json", "Improved V1": REPORT_DIR.parent / "improved_v1_results.json", "Improved V2": results_path}, REPORT_DIR / "model_comparison.csv")
    LOGGER.info("Saved checkpoint: %s", checkpoint_path)
    LOGGER.info("Saved results and plots: %s", REPORT_DIR)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not initialize from ImageNet weights")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(parse_args())
