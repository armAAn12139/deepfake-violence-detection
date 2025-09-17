import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocessing import detect_and_crop_faces

# Load models once
deepfake_model = load_model('models/deepfake_model.h5')
violence_model = load_model('models/violence_model.h5')
emotion_model = load_model('models/emotion_model.h5')

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Read image and resize + normalize for model inference
    """
    import cv2
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0)  # Add batch dim

def predict_deepfake(image_path):
    img = preprocess_image(image_path)
    prediction = deepfake_model.predict(img)[0][0]
    label = "Fake" if prediction > 0.5 else "Real"
    return {"label": label, "confidence": float(prediction)}

def predict_violence(image_path):
    img = preprocess_image(image_path)
    prediction = violence_model.predict(img)[0]
    labels = ["Physical Assault", "Weapon Threat", "Verbal Abuse", "None"]
    predicted_label = labels[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return {"label": predicted_label, "confidence": confidence}

def predict_emotion(image_path):
    img = preprocess_image(image_path)
    prediction = emotion_model.predict(img)[0]
    labels = ["Happy", "Angry", "Neutral", "Sad", "Surprised"]
    predicted_label = labels[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return {"label": predicted_label, "confidence": confidence}
