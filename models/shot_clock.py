from pathlib import Path
from .detector_base import BaseDetector
import cv2
import numpy as np
from config.settings import DETECTION_THRESHOLDS, VIOLATION_SETTINGS

class ShotClockDetector(BaseDetector):
    def __init__(self, models=None):
        super().__init__(models)
        
        if not models:
            weights_dir = Path(__file__).parent / "weights"
            weights_dir.mkdir(exist_ok=True)
            
            model_paths = {
                'detection': str(weights_dir / 'best_1_27.pt')
            }
            self._load_models(model_paths)
            
        self.distance_threshold = DETECTION_THRESHOLDS['interaction_distance']
        self.violation_time_threshold = VIOLATION_SETTINGS['shot_clock']
        self.touch_start_times = []
        self.frame_count = 0
        
    def euclidean_distance(self, boxA, boxB):
        centerA = ((boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2)
        centerB = ((boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2)
        return np.sqrt((centerA[0] - centerB[0])**2 + (centerA[1] - centerB[1])**2)
        
    def process_frame(self, frame):
        results = self.models['detection'](frame, verbose=False, conf=0.5)
        return results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []
        
    def check_violation(self, frame, detections):
        """Check for shot clock violations based on player detections."""
        self.frame_count += 1
        
        if len(detections) == 0:
            return False
            
        # Check for close interactions between players
        interaction_detected = False
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                if self.euclidean_distance(detections[i], detections[j]) < self.distance_threshold:
                    interaction_detected = True
                    if self.frame_count not in self.touch_start_times:
                        self.touch_start_times.append(self.frame_count)
                    break
            if interaction_detected:
                break
                    
        # Check for shot clock violation
        if len(self.touch_start_times) > 0:
            elapsed_frames = self.frame_count - self.touch_start_times[0]
            elapsed_seconds = elapsed_frames / 30  # Assuming 30 FPS
            
            if elapsed_seconds >= self.violation_time_threshold:
                self.touch_start_times = []  # Reset timer
                return True
                
        return False