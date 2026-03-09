# Face Recognition Security System

A Python-based desktop application for complete face registration, training, and real-time recognition. Includes a modern GUI with camera selection and integrated face detection.

## Features
- **Register New User**: Automatically detect and capture facial images to build a dataset.
- **Train System**: Train the local LBPH (Local Binary Patterns Histograms) face recognizer on the captured dataset.
- **Start Security**: Real-time face recognition and bounding box drawing through the selected camera.
- **Camera Selection**: Auto-detects available webcams (using `pygrabber` on Windows) and provides a dropdown to select the preferred camera for capturing or recognition.

## Requirements

Install the dependencies via `pip`:

```bash
pip install -r requirements.txt
```

### Dependencies Included:
- `opencv-python`: Core OpenCV library for image processing.
- `opencv-contrib-python`: Contains the `face` module needed for `LBPHFaceRecognizer`.
- `numpy`: Required by OpenCV for matrix and array operations.
- `pygrabber`: Used on Windows to fetch actual camera names instead of numerical indices.

> **Note on `pygrabber`:** If you are not on Windows, `pygrabber` might not work. The app has a fallback that will simply list "Camera 0", "Camera 1", etc., if `pygrabber` cannot be loaded.

## How to Run

1. Open a terminal in the project directory.
2. Run the launcher script:
   ```bash
   python launcher.py
   ```
3. A Security Dashboard GUI will open.

### Usage Workflow
1. Select your preferred camera from the dropdown menu.
2. **1. Register New User**: Click this, enter your name, and position yourself in front of the camera. The system will take a series of pictures. Press `q` to finish or let it capture a batch and stop.
3. **2. Train System**: Click this to train the facial recognition model on the newly captured images. Wait for the success message.
4. **3. Start Security**: Click this to launch the live recognition window. It will place a green box around recognized faces (with their name) and a red box around faces it cannot confidently identify. Press `q` to exit the security view.

## Project Structure
- `launcher.py`: The main GUI application.
- `face_auth_system/`: Contains the core logic.
  - `config.py`: Configuration paths and settings.
  - `capture.py`: Handles face detection and image capturing.
  - `train.py`: Handles model training using the captured images.
  - `recognize.py`: Handles real-time face recognition.
- `data/`: Directory where captured user face images are stored (auto-created).
- `models/`: Directory where the trained LBPH model (`lbph_model.xml`) and label mapping (`labels.pkl`) are saved (auto-created).
