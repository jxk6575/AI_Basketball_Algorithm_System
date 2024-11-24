from abc import ABC, abstractmethod
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from config.settings import VIDEO_SETTINGS, MODEL_SETTINGS

class BaseDetector(ABC):
    def __init__(self, models=None):
        self.models = {}
        if models:
            self.models = models
            print("Using pre-initialized models")
        else:
            # Child classes should override this by calling _load_models with proper paths
            pass
    
    def _load_models(self, model_paths):
        """Load models if not provided during initialization"""
        print("\nLoading models...")
        if not model_paths:
            return
            
        for name, path in model_paths.items():
            try:
                model = YOLO(str(path))
                model.overrides['imgsz'] = MODEL_SETTINGS['imgsz']
                self.models[name] = model
                print(f"✓ {name} model loaded successfully")
            except Exception as e:
                print(f"✗ Error loading model {name}: {str(e)}")
                raise RuntimeError(f"Failed to load {name} model")
    
    def load_model(self, name, path):
        if not Path(path).is_absolute():
            path = str(Path(__file__).parent / "weights" / path)
        self.models[name] = YOLO(path)
    
    @abstractmethod
    def process_frame(self, frame):
        """Process a single frame and return detections."""
        pass
    
    @abstractmethod
    def check_violation(self, frame, detections):
        """Check for violations in the processed frame."""
        pass
    
    def draw_debug_info(self, frame, info_dict):
        """Draw debug information on frame."""
        y_offset = 20
        for key, value in info_dict.items():
            cv2.putText(
                frame,
                f"{key}: {value}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            y_offset += 20
        return frame