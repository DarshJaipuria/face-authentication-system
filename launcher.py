import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import threading
from face_auth_system.capture import FaceCapturer
from face_auth_system.train import FaceTrainer
from face_auth_system.recognize import FaceRecognizer

try:
    from pygrabber.dshow_graph import FilterGraph
    HAS_PYGRABBER = True
except ImportError:
    HAS_PYGRABBER = False

class SafetyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Security System")
        self.root.geometry("400x350")
        
        # Initialize modules
        self.capturer = FaceCapturer()
        self.trainer = FaceTrainer()
        self.recognizer = FaceRecognizer()

        # UI Components
        self.title_label = tk.Label(root, text="Security Dashboard", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=10)

        # Camera Selection
        cam_frame = tk.Frame(root)
        cam_frame.pack(pady=5)
        tk.Label(cam_frame, text="Select Camera:").pack(side=tk.LEFT, padx=5)
        
        self.camera_names = self.get_available_cameras()
        self.camera_var = tk.StringVar()
        
        self.cam_dropdown = ttk.Combobox(cam_frame, textvariable=self.camera_var, values=self.camera_names, state="readonly", width=30)
        if self.camera_names:
            self.cam_dropdown.current(0)  # Select the first camera by default
        self.cam_dropdown.pack(side=tk.LEFT)

        self.btn_capture = tk.Button(root, text="1. Register New User", width=25, height=2,
                                     command=self.register_user, bg="#dddddd")
        self.btn_capture.pack(pady=10)

        self.btn_train = tk.Button(root, text="2. Train System", width=25, height=2,
                                   command=self.train_system, bg="#dddddd")
        self.btn_train.pack(pady=10)

        self.btn_recognize = tk.Button(root, text="3. Start Security", width=25, height=2,
                                       command=self.start_security, bg="#ffcccc")
        self.btn_recognize.pack(pady=10)

        self.status_label = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def set_status(self, text):
        self.status_label.config(text=text)

    def get_available_cameras(self):
        if HAS_PYGRABBER:
            try:
                graph = FilterGraph()
                devices = graph.get_input_devices()
                if devices:
                    return devices
            except Exception as e:
                print(f"Error fetching cameras via pygrabber: {e}")
        # Fallback if pygrabber fails or missing
        return [f"Camera {i}" for i in range(5)]

    def register_user(self):
        name = simpledialog.askstring("Input", "Enter Name of Person:")
        if name:
            self.set_status(f"Capturing data for {name}...")
            # Run in thread so UI doesn't freeze before window opens
            # But the capturer uses cv2.imshow which needs to run on main thread potentially or is fine?
            # cv2.imshow generally works fine in threads as long as waitKey is called in same thread.
            # Let's run blocking for simplicity as it opens a new window anyway.
            # Get selected camera index
            selected_cam_idx = self.cam_dropdown.current()
            if selected_cam_idx == -1:
                selected_cam_idx = 0

            success = self.capturer.capture_faces(name, camera_id=selected_cam_idx)
            if success:
                self.set_status(f"Capture complete for {name}.")
                messagebox.showinfo("Success", f"Data captured for {name}.")
            else:
                self.set_status("Capture failed/aborted.")
                messagebox.showerror("Error", "Capture failed or was aborted.")

    def train_system(self):
        self.set_status("Training...")
        # Training might take time, good to thread but for small data blocking is ok.
        # Let's simple block for V1.
        success, msg = self.trainer.train()
        self.set_status(msg)
        if success:
            messagebox.showinfo("Training", msg)
            # Reload recognizer model
            self.recognizer.load_model()
        else:
            messagebox.showerror("Error", msg)

    def start_security(self):
        self.set_status("Security Active...")
        
        # Get selected camera index
        selected_cam_idx = self.cam_dropdown.current()
        if selected_cam_idx == -1:
            selected_cam_idx = 0
            
        self.recognizer.run_recognition(camera_id=selected_cam_idx)
        self.set_status("Security Stopped.")

if __name__ == "__main__":
    root = tk.Tk()
    app = SafetyApp(root)
    root.mainloop()
