from pathlib import Path
from .detector_base import BaseDetector
import cv2
import numpy as np
from config.settings import VIOLATION_SETTINGS

class BackcourtDetector(BaseDetector):
    def __init__(self, models=None):
        super().__init__(models)
        
        if not models:
            weights_dir = Path(__file__).parent / "weights"
            weights_dir.mkdir(exist_ok=True)
            
            model_paths = {
                'detection': str(weights_dir / 'best_1_27.pt')
            }
            self._load_models(model_paths)
        
        self.violation_time_threshold = VIOLATION_SETTINGS['backcourt']
        self.frame_count = 0
        self.backcourt_start = None
        
    def process_frame(self, frame):
        results = self.models['detection'](frame, verbose=False, conf=0.5)
        return results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []
        
    def check_violation(self, frame, detections):
        self.frame_count += 1
        
        if len(detections) == 0:
            return False
            
        # Check for backcourt violation
        if self.backcourt_start is not None:
            elapsed_frames = self.frame_count - self.backcourt_start
            elapsed_seconds = elapsed_frames / 30  # Assuming 30 FPS
            
            if elapsed_seconds >= self.violation_time_threshold:
                self.backcourt_start = None
                return True
                
        return False