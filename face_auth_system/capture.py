import cv2
import os
from datetime import datetime
from . import config

class FaceCapturer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(config.CASCADE_PATH)

    def capture_faces(self, person_name, camera_id=config.CAMERA_ID):
        """
        Runs the face capture loop using cv2.imshow.
        Returns True if successful, False if aborted or error.
        """
        person_name = person_name.strip()
        if not person_name:
            print("Invalid name.")
            return False

        person_dir = os.path.join(config.DATA_DIR, person_name)
        os.makedirs(person_dir, exist_ok=True)

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Cannot open web cam {camera_id}")
            return False

        count = 0
        print(f"Starting capture for {person_name}. Press 'c' to capture, 'q' to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Frame grab failed.")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120)
                )

                # Draw rectangles
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # UI Text
                cv2.putText(frame, f"Captures: {count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, "Press 'c' to capture, 'q' to quit",
                            (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow(f"Capturing: {person_name}", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('c'):
                    if len(faces) == 0:
                        print("No face detected.")
                    else:
                        # Select largest face
                        (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
                        face_crop = gray[y:y+h, x:x+w]
                        face_resized = cv2.resize(face_crop, (200, 200))
                        
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"{person_name}_{ts}.png"
                        path = os.path.join(person_dir, filename)
                        cv2.imwrite(path, face_resized)
                        count += 1
                        print(f"Saved {filename} ({count})")
                
                elif key == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyWindow(f"Capturing: {person_name}")
        
        return count > 0
