#!/usr/bin/env python3
# scripts/remap_labels.py
"""
Optional helper: remap folder names to canonical labels.

Usage:
  python scripts/remap_labels.py

Edit the `MAPPING` dictionary below to map your current folder names -> canonical labels:
  e.g. "potholes" -> "pothole", "street-trash" -> "garbage".
This script will rename folders under data/images accordingly (it moves folders).
"""
import os, shutil

ROOT = "D:/civic-issue-ai/data/images"
MAPPING = {
    # "your_folder_name": "canonical_label",
    # Example:
    # "potholes": "pothole",
    # "street-trash": "garbage",
}

if not os.path.exists(ROOT):
    raise SystemExit("data/images folder not found. Place images under data/images/<label>/")

for src_name, dst_name in MAPPING.items():
    src = os.path.join(ROOT, src_name)
    dst = os.path.join(ROOT, dst_name)
    if not os.path.exists(src):
        print("Skip (not found):", src_name)
        continue
    if os.path.exists(dst):
        print("Destination already exists, merging:", src_name, "->", dst_name)
        # move files
        for f in os.listdir(src):
            shutil.move(os.path.join(src, f), os.path.join(dst, f))
        os.rmdir(src)
    else:
        print("Renaming:", src_name, "->", dst_name)
        shutil.move(src, dst)

print("Done. Verify folders under data/images/")
