from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    try:
        import splitfolders
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'split-folders'. Install it with: "
            "python -m pip install split-folders"
        ) from exc
    splitfolders.ratio(
        str(PROJECT_ROOT / "datasets" / "skin_cancer_3class"),
        output=str(PROJECT_ROOT / "skin_dataset_split"),
        seed=42,
        ratio=(0.7, 0.15, 0.15),
        move=False,
    )


if __name__ == "__main__":
    main()
