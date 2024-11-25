import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch

def verify_model_files():
    weights_dir = Path("models/weights")
    required_weights = [
        "yolov8s-pose.pt",
        "basketballModel.pt",
        "best_1_27.pt"
    ]
    
    for weight in required_weights:
        weight_path = weights_dir / weight
        if not weight_path.exists():
            print(f"✗ Missing required weight file: {weight}")
            return False
    return True

def test_camera():
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✓ Camera working")
                cap.release()
                return True
        print("✗ Camera not working")
        return False
    except Exception as e:
        print(f"✗ Camera error: {str(e)}")
        return False

def test_setup():
    print("\nVerifying setup...")
    models_ok = verify_model_files()
    camera_ok = test_camera()
    
    if models_ok and camera_ok:
        print("\n✓ All checks passed!")
        return True
    else:
        print("\n✗ Some checks failed")
        return False

if __name__ == "__main__":
    test_setup()