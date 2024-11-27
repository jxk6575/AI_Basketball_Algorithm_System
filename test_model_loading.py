import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch
from models.reid.torch_reid import TorchReID

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

def test_reid():
    try:
        reid = TorchReID()
        print("✓ ReID model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ ReID model loading failed: {e}")
        return False

def test_setup():
    print("\nVerifying setup...")
    models_ok = verify_model_files()
    camera_ok = test_camera()
    reid_ok = test_reid()
    
    if models_ok and camera_ok and reid_ok:
        print("\n✓ All checks passed!")
        return True
    else:
        print("\n✗ Some checks failed")
        return False

if __name__ == "__main__":
    test_setup()