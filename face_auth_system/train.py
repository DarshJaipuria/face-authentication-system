import cv2
import os
import numpy as np
import pickle
from . import config

class FaceTrainer:
    def __init__(self):
        pass

    def train(self):
        """
        Trains the LBPH recognizer on data in config.DATA_DIR.
        Returns a tuple (success: bool, message: str)
        """
        if not os.path.exists(config.DATA_DIR):
             return False, "Data directory not found."

        person_folders = [d for d in os.listdir(config.DATA_DIR)
                          if os.path.isdir(os.path.join(config.DATA_DIR, d))]
        person_folders.sort()

        if not person_folders:
            return False, "No person folders found in data/."

        label_map = {}
        faces = []
        labels = []

        print("Loading data...")
        for idx, person in enumerate(person_folders):
            label_map[person] = idx
            folder_path = os.path.join(config.DATA_DIR, person)
            
            for file in os.listdir(folder_path):
                if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                
                img_path = os.path.join(folder_path, file)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    # Ensure resize match
                    img = cv2.resize(img, (200, 200))
                    faces.append(img)
                    labels.append(idx)
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")

        if len(faces) < 2 or len(set(labels)) < 1:
             return False, "Not enough training data. Need at least 2 images."

        faces_np = np.array(faces, dtype=np.uint8)
        labels_np = np.array(labels, dtype=np.int32)

        print(f"Training on {len(faces_np)} images...")
        
        # Train
        recognizer = cv2.face.LBPHFaceRecognizer_create(
             radius=1, neighbors=8, grid_x=8, grid_y=8
        )
        recognizer.train(faces_np, labels_np)

        # Save
        try:
            recognizer.write(config.MODEL_FILE)
            with open(config.LABELS_FILE, "wb") as f:
                pickle.dump(label_map, f)
        except Exception as e:
            return False, f"Failed to save model: {e}"

        return True, f"Training complete! {len(faces)} images, {len(label_map)} people."
