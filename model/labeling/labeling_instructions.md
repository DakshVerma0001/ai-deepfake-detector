# Labeling Instructions (SHORT) — Updated for folder-per-class dataset

## Labels (multi-label allowed per image)
- pothole
- garbage
- water_leak
- broken_light
- tree_fall
- other

## Dataset layout (how images are organized)
We expect images to be stored as:
data/images/pothole/Image_1.jpg
data/images/broken_light/Image_1.jpg
data/images/garbage/Image_1.jpg
data/images/tree_fall/Image_1.jpg
data/images/water_leak/Image_1.jpg


## Attributes (for each label / bbox)
- severity: low, medium, high
- department: Roads, Sanitation, Water, Electrical, Parks, Other  (optional)
- bbox_confidence: 0.0-1.0 (optional numeric)

## Quick rules
- If multiple issues appear in one image, label all relevant classes and provide severity for each (multi-label). If a single image legitimately belongs to multiple class folders, place it in one folder and make sure CSV/annotations are updated to include multi-labels (conversion script will help).
- If image ambiguous, choose the closest label and flag note for reviewer.
- Small issue at critical location (school, hospital, crossing) → increase severity by one level.

## Workflow
1. Preprocessing: run `scripts/generate_classification_csv.py` to create a base CSV mapping images to labels from your folder structure.
2. Labeling: use CVAT (or your annotation tool). Import images from `data/images/`.
3. For detection tasks, annotate bounding boxes in CVAT; export COCO JSON to `data/annotations/`.
4. When finished, save classification CSVs to `data/annotations/` and detection JSONs to `data/annotations/`.

(See severity_rules.md for objective severity definitions and examples per class.)
