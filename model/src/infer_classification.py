# src/infer_classification.py
# Debug-friendly inference: avoids sys.exit() so VSCode debugger doesn't break on SystemExit.
import os
import sys
import glob
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = "models/best_classification.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}. Please run training first (python src/train_classification.py).")
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

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict(model, idx_to_class, image_path, topk=3):
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        confs, idxs = probs.topk(topk, dim=1)
    res = []
    for c,i in zip(confs[0].cpu().tolist(), idxs[0].cpu().tolist()):
        res.append((idx_to_class[i], float(c)))
    return res

def find_fallback_image():
    # prioritise explicit test split entry
    test_split = os.path.join("data", "splits", "test.txt")
    if os.path.exists(test_split):
        with open(test_split, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
            if lines:
                candidate = os.path.normpath(lines[0])
                if os.path.exists(candidate):
                    return candidate
                # search by basename
                base = os.path.basename(candidate)
                found = glob.glob(os.path.join("data","images","**", base), recursive=True)
                if found:
                    return found[0]
    # fallback: first image found under data/images
    imgs = glob.glob(os.path.join("data","images","*","*.jpg"), recursive=True) + \
           glob.glob(os.path.join("data","images","*","*.jpeg"), recursive=True) + \
           glob.glob(os.path.join("data","images","*","*.png"), recursive=True)
    return imgs[0] if imgs else None

def resolve_image_path(img_path):
    if not img_path:
        return None
    # normalize, check absolute/relative
    p = os.path.normpath(img_path)
    if os.path.exists(p):
        return p
    alt = os.path.normpath(os.path.join(os.getcwd(), img_path))
    if os.path.exists(alt):
        return alt
    base = os.path.basename(p)
    matches = glob.glob(os.path.join("data","images","**", base), recursive=True)
    return matches[0] if matches else None

def main():
    # 1) determine raw image candidate
    if len(sys.argv) > 1:
        raw_img = sys.argv[1]
    else:
        raw_img = find_fallback_image()
        if raw_img:
            print("[INFO] No image argument provided. Using fallback image:", raw_img)
        else:
            print("[ERROR] No image argument provided and no images found in data/splits/test.txt or data/images/.")
            print("Usage: python src/infer_classification.py <path/to/image.jpg>")
            # Do NOT call sys.exit here; return gracefully so debugger won't break
            return

    img_path = resolve_image_path(raw_img)
    if img_path is None or not os.path.exists(img_path):
        print("[ERROR] Image not found after resolution attempts.")
        print(" - requested:", raw_img)
        print(" - cwd:", os.getcwd())
        print(" - Try running: python -c \"import glob; print(glob.glob('data/images/**/*.jpg', recursive=True)[:20])\"")
        return

    model, idx_to_class = load_model()
    if model is None:
        # load_model already printed a message
        return

    try:
        print("[INFO] Resolved image path:", img_path)
        preds = predict(model, idx_to_class, img_path, topk=3)
        print("[RESULT]", preds)
    except Exception as e:
        print("[ERROR] Exception during prediction:", e)
# --- compatibility wrapper: simple predict by image path
def predict_image(image_path, topk=3):
    """
    Stable wrapper that returns a list of (label, confidence) for the given image_path.
    It uses the module's load_model() when available, or falls back to predict() variants.
    """
    # Prefer using load_model() if available
    try:
        mdl, idx_to_class = load_model()
    except Exception:
        mdl = None
        idx_to_class = None

    # If predict already accepts (idx_to_class, image_path, topk) use that
    try:
        # try calling predict(idx_to_class, image_path, topk=...)
        return predict(idx_to_class, image_path, topk=topk)
    except TypeError:
        pass
    except Exception:
        # if predict raises other errors, let them propagate
        return predict(idx_to_class, image_path, topk=topk)

    # try predict(image_path, topk=...)
    try:
        return predict(image_path, topk=topk)
    except TypeError:
        pass
    except Exception:
        return predict(image_path, topk=topk)

    # try predict(model, idx_to_class, image_path, topk=...)
    try:
        return predict(mdl, idx_to_class, image_path, topk=topk)
    except Exception:
        # last resort: call predict without additional args
        return predict(image_path)
