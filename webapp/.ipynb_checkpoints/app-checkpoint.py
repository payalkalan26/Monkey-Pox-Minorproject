import os
import io
import json
from typing import Any, Optional, Tuple

from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

# Project settings
APP_TITLE = "Designing a Prediction Model for Detection of Monkeypox"
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
DEFAULT_LABELS = ["Monkeypox", "Non-Monkeypox"]
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

app = Flask(__name__, static_folder="static")
app.secret_key = SECRET_KEY

# Ensure dirs
os.makedirs(UPLOAD_DIR, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _safe_import(module: str):
    try:
        return __import__(module)
    except Exception:
        return None


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


MODEL_CACHE: dict = {"model": None, "framework": None, "labels": DEFAULT_LABELS, "cfg": {}}


def load_model() -> Tuple[Optional[Any], str, list, dict]:
    if MODEL_CACHE.get("model") is not None:
        return MODEL_CACHE["model"], MODEL_CACHE["framework"], MODEL_CACHE["labels"], MODEL_CACHE["cfg"]

    cfg = load_config()
    framework = cfg.get("framework")  # sklearn | keras/tensorflow | pytorch | onnx
    model_path = cfg.get("model_path")
    labels = cfg.get("labels") or DEFAULT_LABELS

    model = None
    if framework and model_path and os.path.exists(os.path.join(BASE_DIR, model_path)):
        full_path = os.path.join(BASE_DIR, model_path)
        if framework == "sklearn":
            joblib = _safe_import("joblib") or _safe_import("sklearn.externals.joblib")
            if joblib:
                try:
                    model = joblib.load(full_path)
                except Exception:
                    model = None
        elif framework in ("keras", "tensorflow"):
            tf = _safe_import("tensorflow")
            if tf:
                try:
                    model = tf.keras.models.load_model(full_path)
                except Exception:
                    model = None
        elif framework == "pytorch":
            torch = _safe_import("torch")
            if torch:
                try:
                    model = torch.jit.load(full_path) if full_path.endswith((".pt", ".pth")) else None
                except Exception:
                    model = None
        elif framework == "onnx":
            ort = _safe_import("onnxruntime")
            if ort:
                try:
                    model = ort.InferenceSession(full_path)
                except Exception:
                    model = None

    MODEL_CACHE.update({"model": model, "framework": framework, "labels": labels, "cfg": cfg})
    return model, framework, labels, cfg


def preprocess_image(img: Image.Image, cfg: dict) -> np.ndarray:
    size = cfg.get("input_size", [224, 224])
    to_rgb = cfg.get("to_rgb", True)
    normalize = cfg.get("normalize", True)
    mean = np.array(cfg.get("mean", [0.485, 0.456, 0.406]))
    std = np.array(cfg.get("std", [0.229, 0.224, 0.225]))

    if to_rgb:
        img = img.convert("RGB")
    img = img.resize((size[1], size[0]))
    arr = np.array(img).astype("float32") / 255.0
    if normalize:
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = (arr - mean) / std
        else:
            arr = (arr - mean.mean()) / std.mean()
    arr = np.transpose(arr, (2, 0, 1)) if arr.ndim == 3 else arr
    arr = np.expand_dims(arr, 0)
    return arr


def predict_image(model: Any, x: np.ndarray, framework: Optional[str], labels: list) -> Tuple[str, float]:
    try:
        if model is None:
            # Demo: naive confidence
            return "Model not configured", 0.0

        if framework == "sklearn":
            x2 = x.reshape((x.shape[0], -1))
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x2)[0]
                idx = int(np.argmax(probs))
                return labels[idx] if idx < len(labels) else str(idx), float(np.max(probs))
            pred = model.predict(x2)[0]
            return str(pred), 1.0

        if framework in ("keras", "tensorflow"):
            probs = model.predict(x, verbose=0)[0]
            if np.ndim(probs) == 0:
                conf = float(probs)
                label = labels[1 if conf < 0.5 else 0]
                return label, float(conf if conf >= 0.5 else 1.0 - conf)
            idx = int(np.argmax(probs))
            return labels[idx] if idx < len(labels) else str(idx), float(np.max(probs))

        if framework == "pytorch":
            import torch
            with torch.no_grad():
                xt = torch.from_numpy(x)
                out = model(xt)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                idx = int(np.argmax(probs))
                return labels[idx] if idx < len(labels) else str(idx), float(np.max(probs))

        if framework == "onnx":
            import onnxruntime as ort
            input_name = model.get_inputs()[0].name
            out = model.run(None, {input_name: x})[0][0]
            if np.ndim(out) == 0:
                conf = float(out)
                label = labels[1 if conf < 0.5 else 0]
                return label, float(conf if conf >= 0.5 else 1.0 - conf)
            idx = int(np.argmax(out))
            return labels[idx] if idx < len(labels) else str(idx), float(np.max(out))

        return "Unsupported framework", 0.0
    except Exception as e:
        return f"Prediction error: {e}", 0.0


@app.route("/")
def home():
    if session.get("user"):
        return redirect(url_for("upload"))
    return redirect(url_for("signin"))


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        org = request.form.get("org", "").strip()
        if not name or not email:
            flash("Please enter your name and email.", "warning")
            return render_template("signin.html", title=APP_TITLE)
        session["user"] = {"name": name, "email": email, "org": org}
        flash("Signed in successfully.", "success")
        return redirect(url_for("upload"))
    return render_template("signin.html", title=APP_TITLE)


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been signed out.", "info")
    return redirect(url_for("signin"))


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if not session.get("user"):
        return redirect(url_for("signin"))

    model, framework, labels, cfg = load_model()
    model_ready = model is not None and framework is not None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Please select an image.", "warning")
            return render_template("upload.html", title=APP_TITLE, model_ready=model_ready)
        if not allowed_file(file.filename):
            flash("Unsupported file type. Upload PNG/JPG/JPEG.", "danger")
            return render_template("upload.html", title=APP_TITLE, model_ready=model_ready)

        filename = secure_filename(file.filename)
        saved_path = os.path.join(UPLOAD_DIR, filename)
        file.save(saved_path)

        with Image.open(saved_path) as img:
            x = preprocess_image(img, cfg)
        label, conf = predict_image(model, x, framework, labels)
        return render_template("result.html", title=APP_TITLE, image_url=url_for("uploaded_file", filename=filename), label=label, confidence=conf)

    return render_template("upload.html", title=APP_TITLE, model_ready=model_ready)


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
