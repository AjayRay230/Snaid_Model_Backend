import onnxruntime as ort
import cv2
import numpy as np
from scipy.special import softmax
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import os
import gdown

# -------------------------------
# Load metadata
# -------------------------------
metadata_path = "data/venomstatus_with_antivenom.csv"
metadata = pd.read_csv(metadata_path)

# Prepare LabelBinarizer
labels_species = metadata['binomial']
lb_species = LabelBinarizer()
lb_species.fit(np.asarray(labels_species))

# -------------------------------
# Download model from Google Drive
# -------------------------------
MODEL_PATH = "model.onnx"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1cLvN64Lh0aFIuB4CkEAh0ko422FPBHCy"
    gdown.download(url, MODEL_PATH, quiet=False)

# -------------------------------
# Load ONNX model
# -------------------------------
ort_session = ort.InferenceSession(MODEL_PATH)


# -------------------------------
# Prediction function
# -------------------------------
def predict_image_int32(image_path, ort_session=ort_session, lb_species=lb_species, height=384, width=384):
    """Predict snake species using ONNX model"""

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    image = image.astype(np.int32)

    # ⚠️ IMPORTANT: add batch dimension
    image = np.expand_dims(image, axis=0)

    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: image})
    logits = outputs[0]
    probs = softmax(logits)

    pred_class = int(np.argmax(probs))
    confidence = float(np.max(probs) * 100)
    species_name = lb_species.classes_[pred_class]

    return pred_class, species_name, confidence


# -------------------------------
# Metadata lookup
# -------------------------------
def get_snake_info(pred_class):
    row = metadata[metadata["class_id"] == pred_class].iloc[0]

    venom_status = "Venomous" if row["venom_status"] == 1 else "Non-Venomous"
    antivenom = row["antivenom Name"]
    habitat = row.get("region", "Not available")

    return venom_status, antivenom, habitat