# config.py
import os
import cv2

# Base Paths (Relative to the launcher/root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Model Files
MODEL_FILE = os.path.join(MODELS_DIR, "lbph_model.xml")
LABELS_FILE = os.path.join(MODELS_DIR, "labels.pkl")

# Camera Settings
CAMERA_ID = 0  # Default webcam

# Face Detection
# Use OpenCV's built-in Haar cascade
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
