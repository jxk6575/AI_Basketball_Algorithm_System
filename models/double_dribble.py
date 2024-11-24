from pathlib import Path
from .detector_base import BaseDetector
import cv2
import numpy as np
import time
from config.settings import DETECTION_THRESHOLDS, BODY_INDICES

class DoubleDribbleDetector(BaseDetector):
    def __init__(self, models=None):
        super().__init__(models)
        
        if not models:
            weights_dir = Path(__file__).parent / "weights"
            weights_dir.mkdir(exist_ok=True)
            
            model_paths = {
                'pose': str(weights_dir / 'yolov8s-pose.pt'),
                'ball': str(weights_dir / 'basketballModel.pt')
            }
            self._load_models(model_paths)
        
        # Initialize parameters
        self.body_index = {
            "left_wrist": BODY_INDICES['left_wrist'],
            "right_wrist": BODY_INDICES['right_wrist']
        }
        self.hold_duration = DETECTION_THRESHOLDS['hold_duration']
        self.hold_threshold = DETECTION_THRESHOLDS['hold_threshold']
        self.dribble_threshold = DETECTION_THRESHOLDS['dribble_threshold']
        self.reset_state()
    
    def reset_state(self):
        # Reference from double_dribble.py
        self.hold_start_time = None
        self.is_holding = False
        self.was_holding = False
        self.prev_x_center = None
        self.prev_y_center = None
        self.prev_delta_y = None
        self.dribble_count = 0
        self.double_dribble_time = None
    
    def process_frame(self, frame):
        # Reference from double_dribble.py lines 102-133
        pose_results = self.models['pose'](frame, verbose=False, conf=0.5)
        pose_annotated_frame = pose_results[0].plot()
        
        try:
            rounded_results = np.round(pose_results[0].keypoints.numpy(), 1)
            left_wrist = rounded_results[0][self.body_index["left_wrist"]]
            right_wrist = rounded_results[0][self.body_index["right_wrist"]]
        except:
            return pose_annotated_frame, False
            
        ball_results = self.models['ball'](frame, verbose=False, conf=0.65)
        return pose_annotated_frame, ball_results
    
    def check_violation(self, frame, detections):
        if self.was_holding and not self.is_holding and self.dribble_count > 1:
            self.double_dribble_time = time.time()
            return True
        return False