# src/train_pilot.py
"""
Train classification on pilot dataset (data/pilot/).
Saves best model to models/pilot_classification.pth
Usage:
  python src/train_pilot.py --epochs 12 --batch 16 --lr 1e-4
"""
import os, time, argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
from torch.utils.data import Dataset
import random

# Simple dataset that reads the pilot splits (one image per line)
class PilotDataset(Dataset):
    def __init__(self, split_txt, transform=None, classes=None):
        with open(split_txt, 'r', encoding='utf-8') as f:
            self.files = [l.strip() for l in f if l.strip()]
        self.transform = transform
        if classes is None:
            # derive classes from data/pilot/images folders
            root = Path("data/pilot/images")
            classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        self.idx_to_class = {i:c for c,i in self.class_to_idx.items()}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        # p example: data/pilot/images/<label>/<file>.jpg or data/pilot/images/<label>/<file>.jpg
        label = Path(p).parent.name
        target = self.class_to_idx[label]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target, p

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.15,0.15,0.15,0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = PilotDataset("data/pilot/splits/train.txt", transform=train_tf)
    val_ds = PilotDataset("data/pilot/splits/val.txt", transform=val_tf, classes=list(train_ds.class_to_idx.keys()))
    num_classes = len(train_ds.class_to_idx)
    print("Classes:", train_ds.class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    # model: resnet18 with pretrained weights (fast)
    model = models.resnet18(weights=None)  # weights=None avoids torchvision warning; we will load pretrained via provided API on first run if you prefer
    # better: load pretrained weights if available (try-catch)
    try:
        # if torchvision >=0.13, use weights API; fallback to pretrained=True otherwise
        from torchvision.models import ResNet18_Weights
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        try:
            model = models.resnet18(pretrained=True)
        except Exception:
            print("Warning: couldn't load pretrained weights. Training from scratch.")

    # replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # freeze backbone initially to avoid catastrophic forgetting
    for name,param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    best_val = 0.0
    os.makedirs("models", exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, targets, _ in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        train_loss = running_loss / total if total>0 else 0.0
        train_acc = correct / total if total>0 else 0.0

        # after 2 epochs, unfreeze full model for fine-tuning
        if epoch == 2:
            print("Unfreezing backbone for fine-tuning.")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/5, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

        # validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, targets, _ in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
        val_acc = val_correct / val_total if val_total>0 else 0.0

        print(f"Epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        scheduler.step()

        # save best model
        if val_acc > best_val:
            best_val = val_acc
            torch.save({'model_state_dict': model.state_dict(),
                        'class_to_idx': train_ds.class_to_idx},
                       os.path.join("models","pilot_classification.pth"))
            print("Saved best model (val_acc {:.4f})".format(best_val))

    print("Training complete. Best val acc:", best_val)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()
    train(args)
