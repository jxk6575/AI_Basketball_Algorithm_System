import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch
import gdown
from config.settings import MODEL_PATHS, MODEL_URLS

def download_model(model_name):
    """Download model from Google Drive if not present"""
    if model_name not in MODEL_URLS:
        return False
        
    weight_path = Path(MODEL_PATHS[model_name])
    if weight_path.exists():
        return True
        
    print(f"Downloading {model_name} model...")
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Convert sharing URL to direct download URL
        gdown.download(MODEL_URLS[model_name], str(weight_path), fuzzy=True)
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def verify_model_files():
    weights_dir = Path("models/weights")
    required_weights = [
        "yolov8s-pose.pt",
        "basketballModel.pt",
        "best_1_27.pt"
    ]
    
    all_present = True
    for weight in required_weights:
        weight_path = weights_dir / weight
        if not weight_path.exists():
            print(f"✗ Missing required weight file: {weight}")
            # Try to download if URL is available
            model_name = weight.replace('.pt', '')
            if download_model(model_name):
                print(f"✓ Successfully downloaded: {weight}")
            else:
                all_present = False
                
    return all_present

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