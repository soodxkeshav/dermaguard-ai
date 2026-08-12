from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from baseline_model import create_model, get_device


# ============================================================
# DERMA GUARD AI — BASELINE TRAINING
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "fitzpatrick17k"
)

IMAGE_DIR = (
    DATASET_DIR
    / "background removed"
)

TRAIN_CSV = DATASET_DIR / "train.csv"
VAL_CSV = DATASET_DIR / "validation.csv"
TEST_CSV = DATASET_DIR / "test.csv"

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# Training configuration
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

RANDOM_SEED = 42

# Windows-safe
NUM_WORKERS = 0


# Our three classes
CLASS_NAMES = [
    "benign",
    "malignant",
    "non-neoplastic",
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=RANDOM_SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET
# ============================================================

class SkinLesionDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.transform = transform

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        # CSV contains path relative to project root
        image_path = (
            PROJECT_ROOT
            / str(row["image_path"])
        )

        # Open image
        image = Image.open(
            image_path
        ).convert("RGB")

        # Get label
        label_name = row[
            "three_partition_label"
        ]

        label = CLASS_TO_INDEX[
            label_name
        ]

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ],
    ),
])


val_test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ],
    ),
])


# ============================================================
# LOAD CSV FILES
# ============================================================

def load_data():

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    val_df = pd.read_csv(
        VAL_CSV
    )

    test_df = pd.read_csv(
        TEST_CSV
    )

    return (
        train_df,
        val_df,
        test_df
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(train_df):

    counts = (
        train_df[
            "three_partition_label"
        ]
        .value_counts()
    )

    print("\nClass counts:")

    for class_name in CLASS_NAMES:

        print(
            f"{class_name}: "
            f"{counts[class_name]}"
        )

    total = len(train_df)

    weights = []

    for class_name in CLASS_NAMES:

        class_count = counts[
            class_name
        ]

        weight = (
            total
            / (
                len(CLASS_NAMES)
                * class_count
            )
        )

        weights.append(weight)

    weights = torch.tensor(
        weights,
        dtype=torch.float32
    )

    print("\nClass weights:")

    for class_name, weight in zip(
        CLASS_NAMES,
        weights
    ):

        print(
            f"{class_name}: "
            f"{weight:.4f}"
        )

    return weights


# ============================================================
# DATA LOADERS
# ============================================================

def create_dataloaders(
    train_df,
    val_df,
    test_df
):

    train_dataset = SkinLesionDataset(
        train_df,
        transform=train_transform
    )

    val_dataset = SkinLesionDataset(
        val_df,
        transform=val_test_transform
    )

    test_dataset = SkinLesionDataset(
        test_df,
        transform=val_test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(dim=1)
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                outputs.argmax(dim=1)
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_model(
    model,
    loader,
    device
):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            predictions = (
                outputs.argmax(dim=1)
                .cpu()
                .numpy()
            )

            all_predictions.extend(
                predictions
            )

            all_labels.extend(
                labels.numpy()
            )

    return (
        np.array(all_labels),
        np.array(all_predictions)
    )


# ============================================================
# TRAINING CURVES
# ============================================================

def save_training_curves(
    history
):

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    plt.figure()

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "DermaGuard AI — Loss"
    )

    plt.legend()

    plt.tight_layout()

    loss_path = (
        REPORT_DIR
        / "baseline_loss_curve.png"
    )

    plt.savefig(
        loss_path,
        dpi=200
    )

    plt.close()


    # Accuracy plot

    plt.figure()

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "DermaGuard AI — Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    accuracy_path = (
        REPORT_DIR
        / "baseline_accuracy_curve.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=200
    )

    plt.close()


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure()

    plt.imshow(cm)

    plt.title(
        "DermaGuard AI — Confusion Matrix"
    )

    plt.colorbar()

    tick_positions = range(
        len(CLASS_NAMES)
    )

    plt.xticks(
        tick_positions,
        CLASS_NAMES,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        tick_positions,
        CLASS_NAMES
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    # Write numbers inside cells
    for i in range(
        len(CLASS_NAMES)
    ):

        for j in range(
            len(CLASS_NAMES)
        ):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    path = (
        REPORT_DIR
        / "baseline_confusion_matrix.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    print("=" * 70)
    print(
        "DERMAGUARD AI — BASELINE RESNET18"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_df, val_df, test_df = (
        load_data()
    )

    print("\nDataset:")
    print(
        f"Training:   {len(train_df)}"
    )

    print(
        f"Validation: {len(val_df)}"
    )

    print(
        f"Test:       {len(test_df)}"
    )

    print(
        f"Total:      "
        f"{len(train_df) + len(val_df) + len(test_df)}"
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    class_weights = (
        calculate_class_weights(
            train_df
        )
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders(
        train_df,
        val_df,
        test_df
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nLoading ResNet18...")

    model = create_model(
        num_classes=3
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    class_weights = (
        class_weights.to(device)
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)

    print(
        f"Model: ResNet18"
    )

    print(
        f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epochs: {NUM_EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Random seed: {RANDOM_SEED}"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    best_val_accuracy = 0.0

    best_model_path = (
        MODEL_DIR
        / "resnet18_baseline_best.pth"
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(
        NUM_EPOCHS
    ):

        print(
            f"\nEpoch "
            f"{epoch + 1}/{NUM_EPOCHS}"
        )

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )
        )

        val_loss, val_accuracy = (
            validate(
                model,
                val_loader,
                criterion,
                device
            )
        )

        history[
            "train_loss"
        ].append(train_loss)

        history[
            "val_loss"
        ].append(val_loss)

        history[
            "train_accuracy"
        ].append(train_accuracy)

        history[
            "val_accuracy"
        ].append(val_accuracy)

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Val Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_accuracy:.4f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = (
                val_accuracy
            )

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    CLASS_NAMES,

                "class_to_index":
                    CLASS_TO_INDEX,

                "image_size":
                    IMAGE_SIZE,

                "batch_size":
                    BATCH_SIZE,

                "learning_rate":
                    LEARNING_RATE,

                "num_epochs":
                    NUM_EPOCHS,

                "best_val_accuracy":
                    best_val_accuracy,

                "random_seed":
                    RANDOM_SEED,
            }

            torch.save(
                checkpoint,
                best_model_path
            )

            print(
                "✓ Best model saved."
            )

    # --------------------------------------------------------
    # Training curves
    # --------------------------------------------------------

    save_training_curves(
        history
    )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    print("\nLoading best checkpoint...")

    checkpoint = torch.load(
        best_model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST EVALUATION")
    print("=" * 70)

    y_true, y_pred = evaluate_model(
        model,
        test_loader,
        device
    )

    test_accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    print(
        f"\nTest Accuracy:  {test_accuracy:.4f}"
    )

    print(
        f"Test Precision: {precision:.4f}"
    )

    print(
        f"Test Recall:    {recall:.4f}"
    )

    print(
        f"Test F1:        {f1:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    save_confusion_matrix(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # Save JSON results
    # --------------------------------------------------------

    results = {

        "model": "ResNet18",

        "dataset": "Fitzpatrick17k",

        "training_samples":
            len(train_df),

        "validation_samples":
            len(val_df),

        "test_samples":
            len(test_df),

        "classes":
            CLASS_NAMES,

        "best_validation_accuracy":
            best_val_accuracy,

        "test_accuracy":
            float(test_accuracy),

        "test_precision_weighted":
            float(precision),

        "test_recall_weighted":
            float(recall),

        "test_f1_weighted":
            float(f1),

        "classification_report":
            report,

        "configuration": {

            "image_size":
                IMAGE_SIZE,

            "batch_size":
                BATCH_SIZE,

            "epochs":
                NUM_EPOCHS,

            "learning_rate":
                LEARNING_RATE,

            "random_seed":
                RANDOM_SEED,

        }
    }

    results_path = (
        REPORT_DIR
        / "baseline_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nBest Validation Accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test F1: "
        f"{f1:.4f}"
    )

    print(
        f"\nModel saved to:"
        f"\n{best_model_path}"
    )

    print(
        f"\nResults saved to:"
        f"\n{results_path}"
    )

    print(
        "\nBaseline model is ready for evaluation."
    )


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()