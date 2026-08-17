from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# ============================================================
# DERMA GUARD AI — IMPROVED MODEL V1
# ============================================================
#
# Improvement over baseline:
# 1. Stronger training augmentation
# 2. ImageNet normalization
# 3. Class-weighted CrossEntropyLoss
# 4. Best checkpoint selected using validation Macro-F1
# 5. Detailed per-class evaluation
#
# IMPORTANT:
# Same train / validation / test splits as baseline.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "datasets" / "fitzpatrick17k"
IMAGE_DIR = DATASET_DIR / "background removed"

TRAIN_CSV = DATASET_DIR / "train.csv"
VAL_CSV = DATASET_DIR / "validation.csv"
TEST_CSV = DATASET_DIR / "test.csv"

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "resnet18_improved_v1_best.pth"
RESULTS_PATH = REPORT_DIR / "improved_v1_results.json"

IMAGE_PATH_COLUMN = "image_path"
TARGET_COLUMN = "three_partition_label"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 0.0001
RANDOM_SEED = 42

NUM_WORKERS = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_NAMES = [
    "benign",
    "malignant",
    "non-neoplastic",
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

INDEX_TO_CLASS = {
    index: name
    for name, index in CLASS_TO_INDEX.items()
}


# ============================================================
# TRANSFORMS
# ============================================================

# Training:
# More variation than baseline while avoiding aggressive
# color manipulation because skin-tone analysis is important
# later in the project.

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.80, 1.0)
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# Validation/Test:
# NO augmentation.
# Evaluation must remain deterministic.

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# DATASET
# ============================================================

class FitzpatrickDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = PROJECT_ROOT / row[IMAGE_PATH_COLUMN]

        image = Image.open(image_path).convert("RGB")

        label_name = row[TARGET_COLUMN]

        if label_name not in CLASS_TO_INDEX:
            raise ValueError(
                f"Unknown class label: {label_name}"
            )

        label = CLASS_TO_INDEX[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("DERMAGUARD AI — IMPROVED MODEL V1")
print("=" * 70)

print(f"\nDevice: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


print("\nDataset:")
print(f"Training:   {TRAIN_CSV}")
print(f"Validation: {VAL_CSV}")
print(f"Test:       {TEST_CSV}")


train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)


print("\nDataset sizes:")
print(f"Training samples:   {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples:       {len(test_df)}")
print(f"Total samples:      {len(train_df) + len(val_df) + len(test_df)}")


# ============================================================
# VERIFY CLASSES
# ============================================================

all_labels = sorted(
    set(train_df[TARGET_COLUMN])
    | set(val_df[TARGET_COLUMN])
    | set(test_df[TARGET_COLUMN])
)

print("\nClasses:")
print(all_labels)

if set(all_labels) != set(CLASS_NAMES):
    raise ValueError(
        f"Unexpected classes detected: {all_labels}"
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print("\nTraining:")
print(train_df[TARGET_COLUMN].value_counts())

print("\nValidation:")
print(val_df[TARGET_COLUMN].value_counts())

print("\nTest:")
print(test_df[TARGET_COLUMN].value_counts())


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = FitzpatrickDataset(
    train_df,
    transform=train_transform
)

val_dataset = FitzpatrickDataset(
    val_df,
    transform=eval_transform
)

test_dataset = FitzpatrickDataset(
    test_df,
    transform=eval_transform
)


# ============================================================
# DATA LOADERS
# ============================================================

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


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = (
    train_df[TARGET_COLUMN]
    .value_counts()
    .reindex(CLASS_NAMES)
)

print("\nClass counts:")
print(class_counts)


# Inverse-frequency style weighting.
#
# More weight is given to classes with fewer samples.

total_samples = len(train_df)

class_weights = total_samples / (
    len(CLASS_NAMES) * class_counts.values
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

print("\nClass weights:")

for class_name, weight in zip(
    CLASS_NAMES,
    class_weights.cpu().numpy()
):
    print(f"{class_name}: {weight:.4f}")


# ============================================================
# MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING RESNET18")
print("=" * 70)

try:

    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(
        weights=weights
    )

except AttributeError:

    # Compatibility fallback for older torchvision.

    model = models.resnet18(
        pretrained=True
    )


# Replace final classification layer.

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)

model = model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader):

    model.train()

    running_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

    epoch_loss = (
        running_loss / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )

    return (
        epoch_loss,
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    loss = (
        running_loss / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )

    return (
        loss,
        accuracy,
        precision,
        recall,
        f1,
        all_labels,
        all_predictions
    )


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

print(f"Model: ResNet18")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Random seed: {RANDOM_SEED}")
print(f"Device: {DEVICE}")

print("\nImprovement V1:")
print("- Random resized crop")
print("- Random horizontal flip")
print("- Small random rotation")
print("- ImageNet normalization")
print("- Class-weighted loss")
print("- Best checkpoint selected by validation Macro-F1")


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "=" * 70)
print("STARTING IMPROVED V1 TRAINING")
print("=" * 70)


best_val_f1 = -1.0
best_epoch = 0

history = []


for epoch in range(EPOCHS):

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_loss, train_acc, train_precision, train_recall, train_f1 = (
        train_one_epoch(
            model,
            train_loader
        )
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    (
        val_loss,
        val_acc,
        val_precision,
        val_recall,
        val_f1,
        _,
        _
    ) = evaluate(
        model,
        val_loader
    )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler.step(val_f1)

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_acc:.4f}"
    )

    print(
        f"Train Macro-F1: {train_f1:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: {val_acc:.4f}"
    )

    print(
        f"Val Macro-F1: {val_f1:.4f}"
    )

    print(
        f"Learning Rate: "
        f"{optimizer.param_groups[0]['lr']:.7f}"
    )

    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_macro_f1": train_f1,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "learning_rate":
            optimizer.param_groups[0]["lr"]
    })

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        best_epoch = epoch + 1

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    CLASS_NAMES,

                "image_size":
                    IMAGE_SIZE,

                "best_val_macro_f1":
                    best_val_f1,

                "best_epoch":
                    best_epoch,

                "random_seed":
                    RANDOM_SEED
            },
            MODEL_PATH
        )

        print(
            "✓ Best model saved."
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST CHECKPOINT")
print("=" * 70)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"Best epoch: {checkpoint['best_epoch']}"
)

print(
    f"Best validation Macro-F1: "
    f"{checkpoint['best_val_macro_f1']:.4f}"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


(
    test_loss,
    test_accuracy,
    test_macro_precision,
    test_macro_recall,
    test_macro_f1,
    test_labels,
    test_predictions
) = evaluate(
    model,
    test_loader
)


print(
    f"\nTest Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)

print(
    f"Test Macro Precision: "
    f"{test_macro_precision:.4f}"
)

print(
    f"Test Macro Recall: "
    f"{test_macro_recall:.4f}"
)

print(
    f"Test Macro F1: "
    f"{test_macro_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    test_labels,
    test_predictions,
    target_names=CLASS_NAMES,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    pd.DataFrame(
        cm,
        index=[
            f"Actual {name}"
            for name in CLASS_NAMES
        ],
        columns=[
            f"Pred {name}"
            for name in CLASS_NAMES
        ]
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "model": "ResNet18 Improved V1",

    "device": str(DEVICE),

    "gpu": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    ),

    "dataset": {
        "training_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
        "total_samples":
            len(train_df)
            + len(val_df)
            + len(test_df)
    },

    "configuration": {
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "random_seed": RANDOM_SEED
    },

    "best_epoch": best_epoch,

    "best_validation_macro_f1":
        float(best_val_f1),

    "test": {
        "loss": float(test_loss),
        "accuracy": float(test_accuracy),
        "macro_precision":
            float(test_macro_precision),
        "macro_recall":
            float(test_macro_recall),
        "macro_f1":
            float(test_macro_f1)
    },

    "confusion_matrix":
        cm.tolist(),

    "history": history
}


with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("IMPROVED V1 TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nBest Validation Macro-F1: "
    f"{best_val_f1:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Test Macro-F1: "
    f"{test_macro_f1:.4f}"
)

print(
    f"\nModel saved to:"
    f"\n{MODEL_PATH}"
)

print(
    f"\nResults saved to:"
    f"\n{RESULTS_PATH}"
)

print("\nReady for baseline vs improved comparison.")