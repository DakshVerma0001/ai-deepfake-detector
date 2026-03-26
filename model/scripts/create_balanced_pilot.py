#!/usr/bin/env python3
# scripts/create_balanced_pilot.py
"""
Create a balanced pilot dataset from data/annotations/classification_all.csv.

Outputs:
  data/pilot/images/<label>/*.jpg    (copied)
  data/pilot/annotations/classification_pilot.csv
  data/pilot/splits/train.txt / val.txt / test.txt
  data/pilot/annotations/images_coco.json  (COCO images list skeleton)

Usage:
  python scripts/create_balanced_pilot.py --per_class 100 --val_ratio 0.15 --test_ratio 0.15

Notes:
- If a class has fewer images than per_class, all images are used.
- Splits are stratified by primary label (approx).
"""
import os, csv, argparse, random, shutil, json
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--per_class", type=int, default=100, help="Max images per class in pilot")
parser.add_argument("--val_ratio", type=float, default=0.15)
parser.add_argument("--test_ratio", type=float, default=0.15)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
IN_CSV = "D:/civic-issue-ai/data/annotations/classification_all.csv"
PILOT_ROOT = "D:/civic-issue-ai/data/pilot"
PILOT_IMG_ROOT = os.path.join(PILOT_ROOT, "images")
ANNOT_DIR = os.path.join(PILOT_ROOT, "annotations")
SPLITS_DIR = os.path.join(PILOT_ROOT, "splits")
os.makedirs(PILOT_IMG_ROOT, exist_ok=True)
os.makedirs(ANNOT_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)

# read csv
rows = []
with open(IN_CSV, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for rr in r:
        rows.append((rr['image_path'], rr['labels']))

class_to_imgs = defaultdict(list)
for img_path, labels in rows:
    primary = labels.split(';')[0].strip()
    class_to_imgs[primary].append((img_path, labels))

# sample per class
pilot_rows = []
for cls, imgs in class_to_imgs.items():
    random.shuffle(imgs)
    chosen = imgs[:args.per_class]
    print(f"Class {cls}: {len(imgs)} available, selecting {len(chosen)}")
    # copy into pilot/images/<cls>/
    dest_dir = os.path.join(PILOT_IMG_ROOT, cls)
    os.makedirs(dest_dir, exist_ok=True)
    for img_path, labels in chosen:
        fname = os.path.basename(img_path)
        src = img_path
        dst = os.path.join(dest_dir, fname)
        # if source uses relative path, ensure path exists; otherwise try project-root join
        if not os.path.exists(src):
            alt = os.path.join(os.getcwd(), src)
            if os.path.exists(alt):
                src = alt
        if not os.path.exists(src):
            print("Warning: source not found, skipping:", img_path)
            continue
        shutil.copy2(src, dst)
        pilot_rows.append((os.path.join("data","pilot","images",cls,fname), labels))

# write classification_pilot.csv
csv_out = os.path.join(ANNOT_DIR, "classification_pilot.csv")
with open(csv_out, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(["image_path", "labels"])
    for r in pilot_rows:
        w.writerow(r)
print("Wrote", csv_out, "rows:", len(pilot_rows))

# create splits (stratified by primary label)
class_to_pilot = defaultdict(list)
for img_path, labels in pilot_rows:
    primary = labels.split(';')[0].strip()
    class_to_pilot[primary].append(img_path)

train, val, test = [], [], []
for cls, imgs in class_to_pilot.items():
    random.shuffle(imgs)
    n = len(imgs)
    n_val = max(1, int(n * args.val_ratio))
    n_test = max(1, int(n * args.test_ratio))
    n_train = n - n_val - n_test
    if n_train < 1 and n > 0:
        n_train = 1
        if n_val>0: n_val -= 1
        elif n_test>0: n_test -= 1
    train += imgs[:n_train]
    val += imgs[n_train:n_train+n_val]
    test += imgs[n_train+n_val:]

# write splits
for name, lst in [("train.txt", train), ("val.txt", val), ("test.txt", test)]:
    p = os.path.join(SPLITS_DIR, name)
    with open(p, 'w', encoding='utf-8') as f:
        for item in sorted(lst):
            f.write(item + "\n")
    print("Wrote", p, len(lst))

# create COCO images skeleton (no annotations)
images = []
img_id = 1
for cls, imgs in class_to_pilot.items():
    for img in imgs:
        fname = os.path.basename(img)
        # COCO image file_name should be relative to pilot/images root or how your annotator wants it.
        images.append({
            "id": img_id,
            "file_name": os.path.join(cls, fname).replace("\\","/"),
            "width": 0,
            "height": 0
        })
        img_id += 1

coco = {"images": images, "annotations": [], "categories": [{"id":1,"name":"placeholder"}]}
coco_out = os.path.join(ANNOT_DIR, "images_coco.json")
with open(coco_out, 'w', encoding='utf-8') as f:
    json.dump(coco, f, indent=2)
print("Wrote COCO images skeleton to", coco_out, "with", len(images), "images")

print("Pilot dataset ready in", PILOT_ROOT)
