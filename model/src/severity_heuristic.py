# src/severity_heuristic.py
"""
Heuristic severity estimation using model softmax confidence + Grad-CAM area.

Usage (CLI):
  python src/severity_heuristic.py [<image_path>]

Returns (dict):
  {
    "predicted": (label, prob),
    "cam_area": 0.12,   # fraction of heatmap above threshold or None
    "severity": "low"   # low/medium/high
  }

Note: model expected at models/pilot_classification.pth
"""
import os, sys, glob, random
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models

MODEL_PATH = "models/pilot_classification.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- model loader
def load_model():
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Model not found:", MODEL_PATH)
        return None, None
    ck = torch.load(MODEL_PATH, map_location=DEVICE)
    class_to_idx = ck.get('class_to_idx', {})
    idx_to_class = {int(v):k for k,v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    model = models.resnet18(weights=None)
    # try modern API
    try:
        from torchvision.models import ResNet18_Weights
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        try:
            model = models.resnet18(pretrained=True)
        except Exception:
            # fall back to uninitialized
            pass
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ck['model_state_dict'])
    model = model.to(DEVICE).eval()
    return model, idx_to_class

# --- preprocessing
_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# --- simple grad-cam approx (no backward)
def compute_cam_area(model, image_path):
    features = []
    def hook(m,i,o):
        features.append(o.detach())
    try:
        final_conv = model.layer4[1].conv2
        h = final_conv.register_forward_hook(hook)
    except Exception:
        return None, None
    img = Image.open(image_path).convert('RGB')
    x = _preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
    if not features:
        try:
            h.remove()
        except Exception:
            pass
        return None, None
    feat = features[0].squeeze(0)   # C,H,W
    weights = feat.mean(dim=(1,2)).cpu().numpy()
    cam = np.zeros(feat.shape[1:], dtype=np.float32)
    for i,w in enumerate(weights):
        cam += w * feat[i].cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))
    cam = (cam - cam.min()) / (cam.max()-cam.min() + 1e-8)
    # proportion of pixels above threshold (0.5)
    area_fraction = float((cam > 0.5).sum()) / (cam.size)
    try:
        h.remove()
    except Exception:
        pass
    return area_fraction, cam

# --- heuristic
def heuristic_severity(pred_prob, cam_area):
    if pred_prob is None:
        return "low"
    if pred_prob < 0.35:
        return "low"
    if cam_area is None:
        # fallback to prob-only
        if pred_prob > 0.8:
            return "high"
        if pred_prob > 0.5:
            return "medium"
        return "low"
    if cam_area > 0.18 or (pred_prob > 0.85 and cam_area > 0.08):
        return "high"
    if cam_area > 0.06 or (pred_prob > 0.55 and pred_prob <= 0.85):
        return "medium"
    return "low"

# --- top-level function for API usage
def estimate_severity(image_path=None, model=None, idx_to_class=None):
    # choose an image if not provided
    if image_path is None:
        candidates = glob.glob("data/pilot/images/*/*.*")
        if not candidates:
            return {"error": "no images found in data/pilot/images"}
        image_path = random.choice(candidates)

    # resolve path variations
    if not os.path.exists(image_path):
        # try join with cwd
        alt = os.path.join(os.getcwd(), image_path)
        if os.path.exists(alt):
            image_path = alt
        else:
            # try find by basename
            base = os.path.basename(image_path)
            found = glob.glob(os.path.join("data","pilot","images","**", base), recursive=True)
            if found:
                image_path = found[0]
            else:
                return {"error": f"image not found: {image_path}"}

    # load model if not passed
    if model is None or idx_to_class is None:
        model, idx_to_class = load_model()
        if model is None:
            return {"error": "model not found"}

    # get top-1 prob and label
    img = Image.open(image_path).convert('RGB')
    x = _preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        top_p, top_idx = probs.topk(1, dim=1)
        top_p = float(top_p[0,0].cpu().numpy())
        top_label = idx_to_class[int(top_idx[0,0].cpu().numpy())]

    cam_area, cam = compute_cam_area(model, image_path)
    severity = heuristic_severity(top_p, cam_area)
    return {"image_path": image_path, "predicted": (top_label, top_p), "cam_area": cam_area, "severity": severity}

# --- CLI behavior
if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else None
    res = estimate_severity(img)
    print(res)
    # If you want to save the cam overlay, you can extend this script to save the cam image.
