# Camera Viewer - Object Detection & Tracking
A GUI application using PyQt5 (or PySide6) for camera viewing, real-time object detection, and tracking with AI models (YOLO, Faster R-CNN, DETR).


## Reference Papers
- Paper You Only Look Once: Unified, Real-Time Object Detection: https://arxiv.org/abs/1506.02640
- Paper Fast R-CNN: https://arxiv.org/abs/1504.08083
- Paper End-to-End Object Detection with Transformers: http://arxiv.org/abs/2005.12872

## Overview
### Project Goal:
Develop a desktop GUI application to support Model Cloning, Customization, and Finetuning on three major object detection models (YOLO, Faster R-CNN, DETR), enabling:

- Cloning pre-trained models to local environments.
- Customizing architecture or configurations (e.g., anchor boxes, head layers).
- Finetuning on custom datasets for specific detection tasks.
- Visual comparison of results with original pretrained models on camera streams and video files.

### Supported Models
🔹 YOLO (You Only Look Once)

- Real-time, single-stage detection.
- Strong speed with acceptable accuracy.
- Easily customizable for anchor boxes and output heads.

🔹 Faster R-CNN

- Two-stage, region proposal and classification refinement.
- High detection accuracy, suitable for smaller-scale deployments.
- Flexible for backbone replacement and ROI adjustments.

🔹 DETR (Detection Transformer)

- Transformer-based end-to-end detection model.
- Uses self-attention for global feature learning.
- Good for cluttered scenes, extensible for different output heads.

Key Objectives
1. Clone pretrained YOLO, Faster R-CNN, and DETR models for local use.
2. Customize model configurations to match dataset and hardware constraints.
3. Finetune each model on your custom dataset (e.g., brain tumor, traffic, industrial defect).
4. Compare finetuned model performance vs. original pretrained models on accuracy, mAP, inference speed, and resource usage.
5. Visualize real-time inference results on a camera stream to evaluate practical performance.



## Demo
<!-- Nếu có demo gif, đặt vào `resources/demo.gif`, nếu chưa có có thể bỏ hoặc thêm sau -->

## Project Structure

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

## How to Run
1.  Run Locally
Install dependencies:
```
bash

pip install -r requirements.txt
```

- Configuration: Edit parameters in config/config.yaml (model paths, GUI settings, camera source, etc.).

- Run the application:


```
bash

python -B src/main.py --config config/config.yaml
```

2. Run with Docker
- Build Docker image:
```
bash

docker build -t camera-viewer .
```

- Run Docker container:
```
bash

docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/app \
    camera-viewer
```


- Notes:

For Windows, use X server (Xming, VcXsrv) and adjust DISPLAY.

- On Linux, allow X server connections:
```
bash
xhost +local:docker
```

## Model Achievements
- YOLO Custom Model (brain tumor dataset):
- Validation Accuracy: 96.2%
- mAP@0.5: 0.93
- Precision: 0.94
- Recall: 0.92
- Supports real-time camera streams & offline videos.
- Supports YOLO, Faster R-CNN, DETR for flexible deployment and research.
- Results based on provided sample datasets. Performance may vary with different datasets and - configurations.

## Main Components
- src/main.py: Entry point for GUI & training.
- src/object_detection/: AI detection & processing modules.
- src/model/: YOLO, Faster R-CNN, DETR models.
- src/ui/: PyQt5/PySide6 user interface components.
- resources/: Icons, images, fonts.
- tests/: Unit tests.


## Technologies Used
- Python 3.x
- PyQt5 / PySide6
- OpenCV
- PyTorch / TensorFlow
- YOLO, Faster R-CNN, DETR
Docker

## License
This project is licensed under the MIT License.

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

For questions, issues, or suggestions, please create an Issue or contact me via email.