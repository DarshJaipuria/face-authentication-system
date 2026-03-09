import cv2
import os
import pickle
from . import config

class FaceRecognizer:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_map = {}
        self.id_to_name = {}
        self.model_loaded = False
        self.load_model()
        self.face_cascade = cv2.CascadeClassifier(config.CASCADE_PATH)

    def load_model(self):
        if os.path.exists(config.MODEL_FILE) and os.path.exists(config.LABELS_FILE):
            try:
                self.recognizer.read(config.MODEL_FILE)
                with open(config.LABELS_FILE, "rb") as f:
                    self.label_map = pickle.load(f)
                self.id_to_name = {v: k for k, v in self.label_map.items()}
                self.model_loaded = True
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return False

    def run_recognition(self, camera_id=config.CAMERA_ID):
        """
        Runs the recognition loop using cv2.imshow.
        """
        if not self.model_loaded:
            if not self.load_model():
                print("Model not loaded/found. Please train first.")
                return

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Cannot open webcam.")
            return

        print("Starting recognition... Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120)
                )

                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_gray = cv2.resize(roi_gray, (200, 200))

                    label_id, confidence = self.recognizer.predict(roi_gray)
                    
                    # Confidence: 0 is perfect match, higher is worse.
                    # Usually < 50 is good, > 80 is unreliable.
                    THRESHOLD = 70.0 
                    name = self.id_to_name.get(label_id, "Unknown")
                    
                    if confidence > THRESHOLD:
                        display_name = "Unknown"
                        color = (0, 0, 255) # Red
                    else:
                        display_name = f"{name}"
                        color = (0, 255, 0) # Green

                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{display_name} ({confidence:.1f})", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                cv2.imshow("Security System - Face Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
