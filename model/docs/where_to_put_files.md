# Where to put files (updated for folder-per-class)

- Raw images (organized by class): `data/images/<label_name>/*.jpg`
  - e.g., `data/images/pothole/Image_1.jpg`
- Converted classification CSV: `data/annotations/classification_all.csv`
- Splits (text files listing filenames relative to data/images/<class>/): `data/splits/train.txt`, `val.txt`, `test.txt`
- Detection exports (COCO JSON): `data/annotations/instances_train.json`, `instances_val.json`, `instances_test.json`
- Example small CSV for labelers/devs: `examples/example_annotations.csv`
