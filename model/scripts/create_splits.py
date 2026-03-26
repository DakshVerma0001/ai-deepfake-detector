#!/usr/bin/env python3
"""
Create train/val/test splits from data/annotations/classification_all.csv.

Outputs:
  data/splits/train.txt
  data/splits/val.txt
  data/splits/test.txt

Splits are image relative paths (one per line).
Default ratio: 70/15/15 (can be changed below).
"""

import os, csv, random
from collections import defaultdict

IN_CSV = "D:/civic-issue-ai/data/annotations/classification_all.csv"
OUT_DIR = "D:/civic-issue-ai/data/splits"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

os.makedirs(OUT_DIR, exist_ok=True)

# read
class_to_imgs = defaultdict(list)
with open(IN_CSV, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for rr in r:
        img = rr['image_path']
        labels = rr['labels']
        # pick first label for stratification (if multi-label, approximate)
        primary = labels.split(';')[0]
        class_to_imgs[primary].append(img)

train, val, test = [], [], []
for cls, imgs in class_to_imgs.items():
    random.shuffle(imgs)
    n = len(imgs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train += imgs[:n_train]
    val += imgs[n_train:n_train+n_val]
    test += imgs[n_train+n_val:]

# if some images missing due to rounding, move them to train
all_assigned = set(train) | set(val) | set(test)
all_images = set()
with open(IN_CSV, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for rr in r:
        all_images.add(rr['image_path'])

unassigned = all_images - all_assigned
train += list(unassigned)

# write
for name, lst in [("train.txt", train), ("val.txt", val), ("test.txt", test)]:
    p = os.path.join(OUT_DIR, name)
    with open(p, 'w', encoding='utf-8') as f:
        for item in sorted(lst):
            f.write(item + "\n")
    print(f"Wrote {p} ({len(lst)} images)")
