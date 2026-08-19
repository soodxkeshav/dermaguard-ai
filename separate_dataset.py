from pathlib import Path
import shutil

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_FILE = PROJECT_ROOT / "datasets" / "fitzpatrick17k" / "fitzpatrick17k.csv"
SOURCE_DIR = PROJECT_ROOT / "datasets" / "fitzpatrick17k" / "background removed"
OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "skin_cancer_3class"

classes = [
    "benign",
    "malignant",
    "non-neoplastic"
]

def main() -> None:
    for class_name in classes:
        (OUTPUT_ROOT / class_name).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_FILE)
    count = 0
    for _, row in df.iterrows():
        label = row["three_partition_label"]
        if label not in classes:
            continue
        filename = f"{row['md5hash']}.jpg"
        source = SOURCE_DIR / filename
        if not source.is_file():
            continue
        destination = OUTPUT_ROOT / label / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        count += 1
        if count % 500 == 0:
            print(f"Copied {count} images")
    print(f"\nDone. Total copied: {count}")


if __name__ == "__main__":
    main()
