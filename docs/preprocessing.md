# DermaGuard AI - Dataset Preprocessing Documentation

This document records the data verification, cleaning, metadata structuring, and splitting pipeline for the **Fitzpatrick17k** dataset.

## 1. Original Dataset Overview
- **Source**: Fitzpatrick17k dataset (`datasets/fitzpatrick17k/fitzpatrick17k.csv`)
- **Original Dataset Size**: 16,577 rows and 9 columns
- **Image Directory**: `datasets/fitzpatrick17k/background removed/`

## 2. Image File Verification
- **Total Image Files Found in Folder**: 16,574 files
- **Supported File Extension**: `.jpg`
- **Filename Convention**: `<md5hash>.jpg`
- **CSV Records with Matching Image**: 16,574 records
- **CSV Records WITHOUT Matching Image**: 3 records (`16,577 - 16,574 = 3`)

## 3. Row Removal & Filtering Criteria
- **Rows Removed**: 3 rows (0.018% of the dataset)
- **Removal Rationale**: Removed exclusively because their corresponding image files could not be located in `background removed/`.
- **Target Label Integrity**: All 16,574 remaining records possess complete target labels (`label`, `three_partition_label`, `nine_partition_label`).

## 4. Treatment of Missing / Unknown Skin-Tone Labels
- **Preservation Policy**: Missing or unknown skin-tone annotations (marked as `-1` in `fitzpatrick_scale` for 565 samples and `fitzpatrick_centaur` for 1,073 samples) were **NOT removed**.
- **Rationale**: Skin-tone metadata is required for post-hoc fairness analysis across Fitzpatrick scale types, but unknown annotations do not invalidate an image for model training.

## 5. Duplicate Analysis
- **Duplicate `md5hash` Count**: 0
- **Duplicate `image_path` Count**: 0
- **Duplicate Rows Count**: 0
- **Status**: No duplicate records found.

## 6. Selected Metadata Fields (`metadata_clean.csv`)
The preprocessed clean metadata DataFrame contains the following 7 standardized fields:
1. `image_path`: Relative path from project root (e.g. `datasets/fitzpatrick17k/background removed/<hash>.jpg`)
2. `md5hash`: Unique image identifier
3. `label`: Fine-grained diagnostic label (114 unique classes)
4. `three_partition_label`: Coarse target classification (`non-neoplastic`, `malignant`, `benign`)
5. `nine_partition_label`: 9-class anatomical/pathological category
6. `fitzpatrick_scale`: Original Fitzpatrick scale score (1–6, or -1 for unknown)
7. `fitzpatrick_centaur`: Original Fitzpatrick Centaur scale score (1–6, or -1 for unknown)

## 7. Data Splitting & Stratification Method
- **Split Ratio**: 70% Training / 15% Validation / 15% Test
- **Fixed Random Seed**: `42`
- **Stratification Method**: Stratified by `three_partition_label` across all splits to preserve exact class ratios.
- **Group Integrity**: Guaranteed 0 hash/path overlap — no `md5hash` or `image_path` appears in more than one split.

## 8. Final Dataset Sizes & Split Verification
- **Total Clean Metadata Rows**: 16,574
- **Training Set (`train.csv`)**: 11,600 samples (70.00%)
- **Validation Set (`validation.csv`)**: 2,485 samples (15.00%)
- **Test Set (`test.csv`)**: 2,489 samples (15.00%)

### Class Distribution (`three_partition_label`) Across Splits:
| Class | Full Metadata | Train Set (70%) | Validation Set (15%) | Test Set (15%) |
|---|---|---|---|---|
| `non-neoplastic` | 12,080 (72.88%) | 8,456 (72.90%) | 1,812 (72.92%) | 1,812 (72.80%) |
| `malignant` | 2,260 (13.64%) | 1,582 (13.64%) | 339 (13.64%) | 339 (13.62%) |
| `benign` | 2,234 (13.48%) | 1,562 (13.47%) | 334 (13.44%) | 338 (13.58%) |

## 9. Generated Artifacts
- [metadata_clean.csv](file:///c:/Users/Keshav%20Sood/OneDrive/Desktop/dermaguard-ai/datasets/fitzpatrick17k/metadata_clean.csv)
- [train.csv](file:///c:/Users/Keshav%20Sood/OneDrive/Desktop/dermaguard-ai/datasets/fitzpatrick17k/train.csv)
- [validation.csv](file:///c:/Users/Keshav%20Sood/OneDrive/Desktop/dermaguard-ai/datasets/fitzpatrick17k/validation.csv)
- [test.csv](file:///c:/Users/Keshav%20Sood/OneDrive/Desktop/dermaguard-ai/datasets/fitzpatrick17k/test.csv)
