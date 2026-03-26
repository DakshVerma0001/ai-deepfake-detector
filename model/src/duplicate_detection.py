# src/duplicate_detection.py
import os, json
import torch, numpy as np
from torchvision import models, transforms
from PIL import Image
from sklearn.cluster import DBSCAN

IMG_LIST = "data/annotations/classification_all.csv"

def load_image_paths(csv_path):
    import csv
    imgs = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for rr in r:
            imgs.append(rr['image_path'])
    return imgs

def extract_embeddings(img_paths, batch=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    emb_list = []
    for p in img_paths:
        img = Image.open(p).convert('RGB')
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            e = model(x).squeeze().cpu().numpy()
        emb_list.append(e / (np.linalg.norm(e)+1e-8))
    return np.vstack(emb_list)

def cluster_embeddings(embs, eps=0.35, min_samples=2):
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embs)
    return clustering.labels_

if __name__ == "__main__":
    paths = load_image_paths(IMG_LIST)
    print("Extracting embeddings for", len(paths), "images...")
    embs = extract_embeddings(paths)
    labels = cluster_embeddings(embs)
    clusters = {}
    for i, l in enumerate(labels):
        clusters.setdefault(int(l), []).append(paths[i])
    # save clusters (label -1 is noise)
    with open("data/annotations/duplicate_clusters.json", "w", encoding='utf-8') as f:
        json.dump(clusters, f, indent=2)
    print("Saved duplicate clusters to data/annotations/duplicate_clusters.json")
