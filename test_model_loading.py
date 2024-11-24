import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch

def verify_model_files():
    try:
        weights_dir = Path("models/weights")
        weights_dir.mkdir(parents=True, exist_ok=True)
        pose_path = weights_dir / "yolov8s-pose.pt"
        # Check pose model with safe loading
        try:
            pose_model = YOLO(pose_path)
            print("✓ Pose model verified")
        except Exception as e:
            print(f"✗ Error loading pose model: {str(e)}")
            return False
        
        # Check basketball model with safe loading
        ball_path = weights_dir / "basketballModel.pt"
        if not ball_path.exists():
            print(f"✗ Basketball model not found at {ball_path}")
            print("Please download from: https://drive.google.com/file/d/1e6HLRuhh1IEmxOFaxHQMxfRqhzD92t3B/view")
            return False
            
        try:
            # Use YOLO loading instead of torch.load
            ball_model = YOLO(str(ball_path))
            # Test if model can process a dummy frame
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = ball_model(dummy_frame, verbose=False)
            print("✓ Basketball model verified")
        except Exception as e:
            print(f"✗ Error loading basketball model: {str(e)}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Model verification failed: {str(e)}")
        return False

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