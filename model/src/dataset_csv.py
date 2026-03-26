# src/dataset_csv.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import csv

class FolderCSVImageDataset(Dataset):
    def __init__(self, split_txt, transform=None, classes=None):
        # split_txt: path to train/val/test txt listing relative image paths
        self.transform = transform
        with open(split_txt, 'r', encoding='utf-8') as f:
            self.files = [l.strip() for l in f.readlines() if l.strip()]
        # create class -> idx map from data/images folders if not provided
        if classes is None:
            root = "D:/civic-issue-ai/data/images"
            classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        self.idx_to_class = {i:c for c,i in self.class_to_idx.items()}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        # p is like data/images/<label>/<img>.jpg
        label = os.path.basename(os.path.dirname(p))
        target = self.class_to_idx[label]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target, p
