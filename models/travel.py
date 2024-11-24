from pathlib import Path
import cv2
import numpy as np
from collections import deque
from .detector_base import BaseDetector
from config.settings import DETECTION_THRESHOLDS, VIOLATION_SETTINGS, VIDEO_SETTINGS

class TravelDetector(BaseDetector):
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
        
        # Initialize parameters from settings
        self.step_threshold = DETECTION_THRESHOLDS['step_threshold']
        self.dribble_threshold = DETECTION_THRESHOLDS['dribble_threshold']
        self.max_ball_not_detected_frames = VIOLATION_SETTINGS['max_ball_not_detected_frames']
        
        # Initialize frame buffer
        self.frame_buffer = deque(maxlen=VIDEO_SETTINGS['frame_buffer_size'])
        self.reset_state()
        
    def reset_state(self):
        self.step_count = 0
        self.dribble_count = 0
        self.prev_ankle_positions = None
        self.ball_not_detected_frames = 0
        self.prev_y_center = None
        self.prev_delta_y = None
        
    def process_frame(self, frame):
        self.frame_buffer.append(frame)
        pose_results = self.models['pose'](frame, verbose=False, conf=0.5)
        ball_results = self.models['ball'](frame, verbose=False, conf=0.65)
        return pose_results, ball_results
        
    def check_violation(self, frame, pose_results, ball_results):
        if len(ball_results) == 0:
            self.ball_not_detected_frames += 1
            if self.ball_not_detected_frames > self.max_ball_not_detected_frames:
                return True
        else:
            self.ball_not_detected_frames = 0
            
        # Check for steps without dribble
        if self.step_count > 2 and self.dribble_count == 0:
            return True
            
        return False