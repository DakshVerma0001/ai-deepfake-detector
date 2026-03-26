#!/usr/bin/env python3
"""
Create a COCO-style images array (no annotations) from folder-per-class images.

Output:
  data/annotations/images_coco.json

This helps tools that expect an images list prior to adding annotations.
"""
import os, json

ROOT = "D:/civic-issue-ai/data/images"
OUT = "D:/civic-issue-ai/data/annotations/images_coco.json"
images = []
img_id = 1
for label in sorted(os.listdir(ROOT)):
    label_dir = os.path.join(ROOT, label)
    if not os.path.isdir(label_dir):
        continue
    for fname in sorted(os.listdir(label_dir)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append({
                "id": img_id,
                "file_name": os.path.join(label, fname),
                "width": 0,
                "height": 0
            })
            img_id += 1

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump({"images": images}, f, indent=2)

print(f"Wrote {OUT} with {len(images)} images")
