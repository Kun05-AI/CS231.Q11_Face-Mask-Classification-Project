#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import uuid
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from skimage.transform import resize
from flask import Flask, render_template, request, redirect, url_for

# ---------------- CONFIG ----------------
IMAGE_SIZE = (128, 128)
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(
    __name__,
    template_folder="static/templates",
    static_folder="static",
)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- LOAD HOG + SVM MODEL ----------------

MODEL_PATH = "models/HOG_8x2_SVM_Optuna.joblib"

data = joblib.load(MODEL_PATH)
clf = data["model"]
le = data["label_encoder"]
CLASSES = list(le.classes_)

print("HOG 8x2 + SVM model loaded. Classes:", CLASSES)

# ---------------- MODEL METRICS (từ classification report SVM) ----------------
# Dựa trên hình mày chụp (Test Accuracy: 0.9899)
MODEL_METRICS = {
    "accuracy": 0.9899,
    "classes": [
        {
            "name": "WithMask",
            "precision": 0.98,
            "recall": 1.00,
            "f1": 0.99,
            "support": 483,
        },
        {
            "name": "WithoutMask",
            "precision": 1.00,
            "recall": 0.98,
            "f1": 0.99,
            "support": 509,
        },
    ],
}

# ---------------- FEATURE EXTRACT (HOG) ----------------
def extract_hog_feature(img_bgr):
    """
    Nhận ảnh BGR full frame, convert -> gray -> resize -> HOG
    trả về vector (1, D)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_resized = resize(gray, IMAGE_SIZE, anti_aliasing=True)

    feat = hog(
        gray_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return feat.reshape(1, -1)


# ---------------- ALLOWED FILE ----------------
def allowed_file(name: str):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------- PREDICT ----------------
def svm_predict_with_prob(X):
    """
    Trả về (proba_vector) dạng (n_classes,)
    - Nếu SVM được train với probability=True: dùng predict_proba
    - Nếu không: dùng decision_function rồi convert softmax / sigmoid
    """
    # Case 1: có predict_proba (khả năng cao vì mày set probability=True)
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[0]

    # Case 2: fallback dùng decision_function
    scores = clf.decision_function(X)[0]  # (C,) hoặc scalar
    scores = np.array(scores)

    if scores.ndim == 0:
        # binary: một score -> sigmoid
        p1 = 1.0 / (1.0 + np.exp(-scores))
        return np.array([1 - p1, p1])

    # multi-class: dùng softmax
    exps = np.exp(scores - scores.max())
    return exps / exps.sum()


def classify_full_image(img_bgr):
    X = extract_hog_feature(img_bgr)  # (1, D)
    proba = svm_predict_with_prob(X)  # (n_classes,)
    idx = int(np.argmax(proba))
    prob = float(proba[idx])

    raw_label = CLASSES[idx]          # "WithMask" / "WithoutMask" ...

    # Map về label hiển thị
    if raw_label in ["WithMask", "Mask"]:
        label = "Mask"
    else:
        label = "No Mask"

    return label, prob


# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    pred_label = None
    pred_prob = None
    result_url = None

    if request.method == "POST":

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file and allowed_file(file.filename):
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None:
                return "Không đọc được ảnh!", 400

            # Predict
            label, prob = classify_full_image(img)

            # Clamp hiển thị <= 99%
            display_prob = min(prob, 0.99)

            # Color for box
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Vẽ bounding toàn ảnh
            h, w = img.shape[:2]
            cv2.rectangle(img, (0, 0), (w, h), color, 4)
            cv2.putText(
                img,
                f"{label}: {display_prob*100:.1f}%",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                color,
                3,
            )

            # Lưu ảnh kết quả
            out_name = f"{uuid.uuid4().hex}.jpg"
            out_path = os.path.join(RESULT_FOLDER, out_name)
            cv2.imwrite(out_path, img)

            result_url = url_for("static", filename=f"results/{out_name}")
            pred_label = label
            pred_prob = display_prob * 100.0  # %

    return render_template(
        "indexSVM.html",
        result_image=result_url,
        pred_label=pred_label,
        pred_prob=pred_prob,
        metrics=MODEL_METRICS,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
