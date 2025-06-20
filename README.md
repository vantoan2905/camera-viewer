# Object Detection & Tracking

## Overview

This project provides a simple camera viewer and a YOLO-like object detection model implemented in PyTorch.  
It includes:

- **Camera Viewer**: View your webcam stream in real-time using Pygame and OpenCV.
- **YOLO Clone**: A simplified YOLO-style neural network for object detection, written from scratch in PyTorch.

## Project Structure

```
object_detect_tracking/
│
├── camera-viewer/
│   └── src/
│       └── main.py         # Camera viewer app (OpenCV + Pygame)
│
├── yolo_clone.py           # YOLO-like model implementation in PyTorch
│
└── README.md               # Project documentation
```

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- Pygame
- PyTorch

Install dependencies with:

```bash
pip install opencv-python pygame torch
```

## Usage

### Camera Viewer

Run the camera viewer:

```bash
python camera-viewer/src/main.py
```

- Press `q` or close the window to exit.

### YOLO Clone

Run the YOLO clone demo:

```bash
python yolo_clone.py
```

- This will create a dummy input and print the shapes of feature maps and predictions.

## Notes

- The YOLO clone is a simplified educational version, not suitable for production.
- To use your own data or train the model, you will need to implement data loading, training, and post-processing steps.

## License

This project is for educational purposes and is licensed under the MIT License - see the LICENSE file for details.