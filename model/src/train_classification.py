# src/train_classification.py
import os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from .dataset_csv import FolderCSVImageDataset

def train(output_dir="models", epochs=10, batch_size=16, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = FolderCSVImageDataset("D:/civic-issue-ai/data/splits/train.txt", transform=train_tf)
    val_ds = FolderCSVImageDataset("D:/civic-issue-ai/data/splits/val.txt", transform=val_tf, classes=list(train_ds.class_to_idx.keys()))
    num_classes = len(train_ds.class_to_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(output_dir, exist_ok=True)
    best_acc = 0.0
    for epoch in range(epochs):
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

        train_loss = running_loss / total
        train_acc = correct / total

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

        print(f"Epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_to_idx': train_ds.class_to_idx
            }, os.path.join(output_dir, "best_classification.pth"))
    print("Training done. Best val acc:", best_acc)

if __name__ == "__main__":
    train()
