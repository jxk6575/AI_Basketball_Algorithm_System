from .detector_base import BaseDetector
import cv2
import numpy as np
from pathlib import Path

class BlockingFoulDetector(BaseDetector):
    def __init__(self, models=None):
        super().__init__(models)
        
        if not models:
            weights_dir = Path(__file__).parent / "weights"
            weights_dir.mkdir(exist_ok=True)
            
            model_paths = {
                'pose': str(weights_dir / 'yolov8s-pose.pt')
            }
            self._load_models(model_paths)
        
        # Body indices for foul detection
        self.body_index = {
            "left_shoulder": 5,
            "right_shoulder": 6,
            "left_hip": 11,
            "right_hip": 12,
            "left_knee": 13,
            "right_knee": 14,
            "left_wrist": 9,
            "right_wrist": 10
        }
        
    def process_frame(self, frame):
        """Process a single frame and return detections"""
        results = self.models['pose'](frame, verbose=False, conf=0.5)
        annotated_frame = results[0].plot()
        
        if len(results) >= 2:
            humans = np.round(results[0].keypoints.numpy(), 1)
            shooter, defender = humans[:2]
            
            # Store body parts for each person
            persons = []
            for human in [shooter, defender]:
                parts = {}
                for part, index in self.body_index.items():
                    try:
                        parts[part] = human[index]
                    except:
                        parts[part] = None
                persons.append(parts)
                
            return persons, annotated_frame
            
        return None, annotated_frame
        
    def check_violation(self, frame, detections):
        """Implementation of abstract method to check for blocking fouls"""
        try:
            persons, _ = self.process_frame(frame)
            if persons and len(persons) >= 2:
                return self.check_foul(persons[1], persons[0])  # defender, shooter
        except Exception as e:
            print(f"Error checking blocking foul: {str(e)}")
        return False
        
    def check_foul(self, defender, shooter):
        """Check if defender is committing a blocking foul"""
        def_parts = ["left_wrist", "right_wrist"]
        shoot_parts = [
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
            "left_wrist",
            "right_wrist",
        ]
        
        for dpart in def_parts:
            for spart in shoot_parts:
                if self.calculate_distance(defender[dpart], shooter[spart]) < 40:
                    return True
        return False
        
    def calculate_distance(self, a, b):
        """Calculate Euclidean distance between two points"""
        if a is None or b is None:
            return float('inf')
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)