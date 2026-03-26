#!/usr/bin/env python3
"""
Generate a classification CSV from a folder-per-class dataset.

Input:
  data/images/<label>/*.jpg

Output:
  data/annotations/classification_all.csv with columns:
    image_path,labels

If you want multi-labels for specific images, create a file:
  data/annotations/multi_label_overrides.csv
with columns:
  image_path,labels
where labels is semicolon-separated.
"""

import os
import csv

ROOT = "D:/civic-issue-ai/data/images"
OUT_DIR = "D:/civic-issue-ai/data/annotations"
OUT_CSV = os.path.join(OUT_DIR, "classification_all.csv")
OVERRIDES = os.path.join(OUT_DIR, "multi_label_overrides.csv")

os.makedirs(OUT_DIR, exist_ok=True)

rows = []
for label in sorted(os.listdir(ROOT)):
    label_dir = os.path.join(ROOT, label)
    if not os.path.isdir(label_dir):
        continue
    for fname in sorted(os.listdir(label_dir)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join("data/images", label, fname)
            rows.append((img_path, label))

# apply overrides (optional)
overrides = {}
if os.path.exists(OVERRIDES):
    with open(OVERRIDES, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for rr in r:
            overrides[rr['image_path']] = rr['labels']

with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(["image_path", "labels"])
    for img_path, label in rows:
        labels = overrides.get(img_path, label)
        w.writerow([img_path, labels])

print(f"Wrote {OUT_CSV} with {len(rows)} rows.")
if overrides:
    print("Applied overrides from", OVERRIDES)