# deploy/inference_api.py
import os
import sys
import importlib
import inspect
from flask import Flask, request, jsonify

# ensure repo root on path so `from src.*` works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try to import helper modules. Import errors will be raised early and visible.
try:
    import src.infer_classification as ic
    from src.severity_heuristic import estimate_severity, load_model as load_severity_model
    from src.routing_simple import route
except Exception as e:
    print("Import error while loading helpers:", e)
    raise

# Create Flask app
app = Flask(__name__)

# Load severity model once at startup (so we don't reload from disk per request)
sev_model = None
sev_idx_to_class = None
try:
    try:
        sev_model, sev_idx_to_class = load_severity_model()
        print("Severity model loaded (inference_api).")
    except Exception as _e:
        print("Warning: could not load severity model at startup:", _e)
        sev_model = None
        sev_idx_to_class = None
except Exception:
    sev_model = None
    sev_idx_to_class = None

# Prepare a generic predict-caller helper that tries multiple signatures.
def call_predict_with_fallbacks(predict_fn, tmp_path):
    """
    Try a set of likely predict(...) signatures and return (preds, used_signature).
    If none work, raise a TypeError with last signature error message.
    """
    last_te = None
    attempts = []
    attempts.append(("predict(image_path, topk=..)", lambda: predict_fn(tmp_path, topk=3)))
    attempts.append(("predict(idx_to_class, image_path, topk=..)", lambda: predict_fn(getattr(ic, "idx_to_class", None), tmp_path, topk=3)))
    attempts.append(("predict(model, idx_to_class, image_path, topk=..)", lambda: predict_fn(getattr(ic, "model", None), getattr(ic, "idx_to_class", None), tmp_path, topk=3)))
    attempts.append(("predict(image_path)", lambda: predict_fn(tmp_path)))
    attempts.append(("predict(idx_to_class, image_path)", lambda: predict_fn(getattr(ic, "idx_to_class", None), tmp_path)))

    for sig, fn in attempts:
        try:
            res = fn()
            return res, sig
        except TypeError as te:
            last_te = te
            continue
        except Exception as e:
            raise RuntimeError(f"predict raised an exception for signature {sig}: {e}") from e

    raise TypeError(f"No working predict signature found. Last TypeError: {last_te}")

@app.route('/infer', methods=['POST'])
def infer():
    # Expect multipart/form-data with 'image' file (and optional text, lat, lon)
    img_file = request.files.get('image')
    text = request.form.get('text', '')
    lat = request.form.get('lat', None)
    lon = request.form.get('lon', None)

    if img_file is None:
        return jsonify({'error': 'no image provided'}), 400

    tmp = "/tmp/infer_tmp.jpg" if os.name != 'nt' else os.path.join(os.environ.get('TEMP','C:\\Temp'),'infer_tmp.jpg')
    try:
        img_file.save(tmp)
    except Exception as e:
        return jsonify({'error': f'failed to save uploaded image: {e}'}), 500

    # --- Classification: prefer stable wrapper `predict_image` if present
    preds_raw = None
    used_sig = None

    try:
        if hasattr(ic, "predict_image"):
            # stable, simple wrapper: predict_image(image_path, topk=3)
            try:
                preds_raw = ic.predict_image(tmp, topk=3)
                used_sig = "predict_image"
            except Exception as e:
                # if wrapper exists but raised, return a clear classification error
                return jsonify({'error': f'predict_image error: {e}'}), 500
        else:
            # fallback to robust signature probing on ic.predict
            if not hasattr(ic, "predict"):
                return jsonify({'error': 'src.infer_classification.predict not found', 'infer_module_exports': sorted([a for a in dir(ic) if not a.startswith("_")])}), 500
            predict_fn = getattr(ic, "predict")
            try:
                preds_raw, used_sig = call_predict_with_fallbacks(predict_fn, tmp)
            except TypeError as e:
                exports = sorted([a for a in dir(ic) if not a.startswith("_")])
                return jsonify({'error': str(e), 'infer_module_exports': exports}), 500
            except RuntimeError as re:
                return jsonify({'error': str(re)}), 500
    except Exception as e:
        return jsonify({'error': f'classification error: {e}'}), 500

    # Normalize preds to list of [label, confidence] if possible
    preds_out = None
    try:
        preds_out = []
        for p in preds_raw:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                lab = str(p[0])
                conf = float(p[1])
                preds_out.append([lab, conf])
            else:
                preds_out.append(p)
    except Exception:
        preds_out = preds_raw

    # Severity: call estimate_severity using preloaded model if available
    try:
        sev_result = estimate_severity(tmp, model=sev_model, idx_to_class=sev_idx_to_class)
    except Exception as e:
        sev_result = {'error': f'severity estimation failed: {e}'}

    # Routing: extract labels for routing
    labels_for_routing = []
    if isinstance(preds_out, list):
        for item in preds_out:
            if isinstance(item, (list,tuple)) and len(item) >= 1:
                labels_for_routing.append(item[0])

    try:
        department = route(labels_for_routing, user_text=text, lat=lat, lon=lon)
    except Exception as e:
        department = f"route_error: {e}"

    result = {
        'predicted': preds_out,
        'predicted_signature_used': used_sig,
        'department': department,
        'severity': sev_result.get('severity') if isinstance(sev_result, dict) else None,
        'severity_details': sev_result
    }
    return jsonify(result)


if __name__ == "__main__":
    # Run from project root: python -m deploy.inference_api
    app.run(debug=True)
