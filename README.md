# Camera Viewer - Object Detection & Tracking

A GUI application using PyQt5/PySide6 for camera viewing, real-time object detection, and tracking with AI models such as YOLO, Faster R-CNN, and DETR.

## 📁 Project Structure

```
camera-viewer/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── run.py
├── config/
│   └── config.yaml
├── resources/
│   └── logo.png
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── app.py
│   ├── object_detection/
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   └── utils.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── detr/
│   │   │   └── detr_model.py
│   │   ├── faster_rcnn/
│   │   │   └── faster_rcnn_model.py
│   │   └── yolo/
│   │       ├── backbone.py
│   │       ├── head.py
│   │       ├── neck.py
│   │       └── yolo_net.py
│   └── ui/
│       ├── __init__.py
│       ├── main_window.py
│       ├── main_window.ui
│       └── widgets/
│           ├── __init__.py
│           ├── video_viewer.py
│           └── result_panel.py
└── tests/
    ├── test_detector.py
    └── test_utils.py
```

## 🚀 How to Run

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Configuration:**  
   Edit parameters in `config/config.yaml` (model paths, GUI settings, etc.).

3. **Run the application:**
    ```bash
    python run.py
    ```

## 🧩 Main Components

- **src/main.py**: Initialize and run the GUI.
- **src/app.py**: Application logic controller.
- **src/object_detection/**: AI processing and object detection.
- **src/model/**: AI models (YOLO, Faster R-CNN, DETR).
- **src/ui/**: User interface (PyQt5/PySide6).
- **resources/**: Images, icons, fonts for the GUI.
- **tests/**: Unit tests for main components.

## 🛠️ Technologies Used

- Python 3.x
- PyQt5 or PySide6
- OpenCV
- Torch/TensorFlow (depending on model)
- AI libraries: YOLO, Faster R-CNN, DETR

## 📄 License

MIT License

---

*For contributions, bug reports, or suggestions, please create an issue or pull request!*