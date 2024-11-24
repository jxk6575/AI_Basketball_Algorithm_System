from pathlib import Path
from .detector_base import BaseDetector
import numpy as np
from config.settings import VIOLATION_SETTINGS, DETECTION_THRESHOLDS

class TenSecondDetector(BaseDetector):
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
        self.violation_time_threshold = VIOLATION_SETTINGS['backcourt']
        self.court_midline = None
        self.touch_start_time = 0
        self.frame_count = 0
        
    def euclidean_distance(self, boxA, boxB):
        centerA = ((boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2)
        centerB = ((boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2)
        return np.sqrt((centerA[0] - centerB[0])**2 + (centerA[1] - centerB[1])**2)
        
    def process_frame(self, frame):
        results = self.models['detection'](frame, verbose=False, conf=0.5)
        return results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []
        
    def check_violation(self, frame, detections):
        """Check for ten second violations based on player detections."""
        self.frame_count += 1
        
        if len(detections) == 0:
            return False
            
        # Set court midline if not set
        if self.court_midline is None:
            self.court_midline = frame.shape[1] // 2
            
        # Check for players crossing midline
        for detection in detections:
            center_x = (detection[0] + detection[2]) / 2
            if center_x < self.court_midline:  # Player in backcourt
                if self.touch_start_time == 0:
                    self.touch_start_time = self.frame_count
                    
                elapsed_frames = self.frame_count - self.touch_start_time
                elapsed_seconds = elapsed_frames / 30  # Assuming 30 FPS
                
                if elapsed_seconds >= 10:  # 10-second violation
                    self.touch_start_time = 0  # Reset timer
                    return True
                    
            else:  # Player crossed to frontcourt
                self.touch_start_time = 0
                
        return False