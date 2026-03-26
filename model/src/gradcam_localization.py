# src/gradcam_localization.py
"""
Robust Grad-CAM overlay script for prototype localization.

Usage:
  python src/gradcam_localization.py <path/to/image> [out_path]

If <path/to/image> is missing or not found, the script will try:
 - first entry of data/splits/test.txt (if exists)
 - first image under data/images/*/*.jpg (fallback)
It prints helpful diagnostics when a file/model is missing.
"""
import os
import glob
import sys
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models

MODEL_PATH = "models/best_classification.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}. Train first (python src/train_classification.py).")
        return None, None
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_to_idx = checkpoint.get('class_to_idx', {})
    idx_to_class = {int(v):k for k,v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE).eval()
    return model, idx_to_class

# Preprocess transform
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Hook container
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.detach())

def find_fallback_image():
    # 1) test split
    test_split = os.path.join("data","splits","test.txt")
    if os.path.exists(test_split):
        with open(test_split, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
            if lines:
                cand = os.path.normpath(lines[0])
                if os.path.exists(cand):
                    return cand
                # search by basename
                base = os.path.basename(cand)
                found = glob.glob(os.path.join("data","images","**", base), recursive=True)
                if found:
                    return found[0]
    # 2) first image under data/images
    imgs = glob.glob(os.path.join("data","images","*","*.jpg"), recursive=True) + \
           glob.glob(os.path.join("data","images","*","*.jpeg"), recursive=True) + \
           glob.glob(os.path.join("data","images","*","*.png"), recursive=True)
    return imgs[0] if imgs else None

def resolve_image_path(img_path):
    if not img_path:
        return None
    p = os.path.normpath(img_path)
    if os.path.exists(p):
        return p
    alt = os.path.normpath(os.path.join(os.getcwd(), img_path))
    if os.path.exists(alt):
        return alt
    # try searching by basename
    base = os.path.basename(p)
    matches = glob.glob(os.path.join("data","images","**", base), recursive=True)
    return matches[0] if matches else None

def gradcam(image_path, target_class=None, out_path='gradcam.jpg'):
    model, idx_to_class = load_model()
    if model is None:
        return False

    # Register hook on last conv layer of resnet18
    features_blobs.clear()
    try:
        final_conv = model.layer4[1].conv2
        final_conv.register_forward_hook(hook_feature)
    except Exception as e:
        print("[WARN] Couldn't attach hook to expected layer:", e)
        return False

    # load image
    img = Image.open(image_path).convert('RGB')
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    # forward
    logits = model(tensor)
    probs = F.softmax(logits, dim=1)
    if target_class is None:
        target = int(probs.argmax(dim=1).item())
    else:
        target = int(target_class)

    # get feature map (channels, h, w)
    if not features_blobs:
        print("[ERROR] No features captured by hook; aborting.")
        return False
    feature = features_blobs[0].squeeze(0)  # (C,H,W)

    # derive channel weights by global average of feature maps (simple, no backward pass)
    weights = feature.mean(dim=(1,2)).cpu().numpy()  # (C,)

    cam = np.zeros(feature.shape[1:], dtype=np.float32)  # H,W
    for i, w in enumerate(weights):
        cam += w * feature[i].cpu().numpy()

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # overlay (resize original to 224)
    orig = cv2.imread(image_path)
    if orig is None:
        print("[ERROR] cv2 couldn't read image (unexpected).")
        return False
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    orig = cv2.resize(orig, (224,224))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.45 * heatmap + 0.55 * orig).astype(np.uint8)
    # save as BGR to disk
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return True

def main():
    # parse args
    if len(sys.argv) > 1:
        raw = sys.argv[1]
    else:
        raw = find_fallback_image()
        if raw:
            print("[INFO] No image argument provided. Using fallback:", raw)
        else:
            print("[ERROR] No image argument and no images found in data/images. Nothing to do.")
            return

    img_path = resolve_image_path(raw)
    if img_path is None or not os.path.exists(img_path):
        print("[ERROR] Image not found after resolution attempts.")
        print(" - requested:", raw)
        print(" - cwd:", os.getcwd())
        print("List first 20 images found by glob:")
        print(glob.glob('data/images/**/*.jpg', recursive=True)[:20])
        return

    out = sys.argv[2] if len(sys.argv) > 2 else "gradcam_out.jpg"
    print("[INFO] Resolved image:", img_path, " -> writing to:", out)
    ok = gradcam(img_path, out_path=out)
    if ok:
        print("[OK] Grad-CAM saved to", out)
    else:
        print("[FAILED] Grad-CAM failed. See messages above.")

if __name__ == "__main__":
    main()
