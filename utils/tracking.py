import cv2
import numpy as np
from collections import deque
from ultralytics.trackers.bot_sort import BOTSORT
from config.settings import TRACKING_SETTINGS

class PlayerTracker:
    def __init__(self, track_buffer=TRACKING_SETTINGS['track_buffer']):
        # Initialize tracking parameters
        self.track_buffer = track_buffer
        self.tracks = {}  # Dictionary to store player tracks
        self.next_id = 0
        
        # Create args object for BOTSORT initialization
        class Args:
            reid_enabled = TRACKING_SETTINGS['reid_enabled']
            appearance_thresh = TRACKING_SETTINGS['appearance_thresh']
            proximity_thresh = TRACKING_SETTINGS['proximity_thresh']
            track_buffer = TRACKING_SETTINGS['track_buffer']
            with_reid = True
            
            # Additional required BOTSORT parameters
            gmc_method = 'sparseOptFlow'
            track_high_thresh = 0.6
            track_low_thresh = 0.1
            new_track_thresh = 0.7
            match_thresh = 0.8
            
        # Initialize BOTSORT with args
        self.tracker = BOTSORT(Args(), frame_rate=30)
        
    def update(self, detections, frame):
        if detections is None or len(detections) == 0:
            return []
            
        # Create YOLO Results format
        class Results:
            def __init__(self, boxes, scores, classes):
                self.xyxy = boxes
                self.conf = scores
                self.cls = classes  # Add class information
        
        # Format detections
        boxes = []
        scores = []
        classes = []
        for det in detections:
            boxes.append(det[:4])  # xyxy coordinates
            scores.append(det[4])  # confidence score
            classes.append(1)  # All detections are people (class 1)
            
        # Create Results object with all required attributes
        results = Results(
            np.array(boxes), 
            np.array(scores),
            np.array(classes)
        )
        
        # Update tracker with Results object
        tracks = self.tracker.update(results)
        return tracks

class BallTracker:
    def __init__(self, max_points=20):
        self.ball_positions = deque(maxlen=max_points)
        self.prev_center = None
        self.dribble_threshold = 18
        
    def update(self, ball_detections):
        if ball_detections is None or len(ball_detections) == 0:
            self.ball_positions.append(None)
            return None
            
        # Get ball center
        bbox = ball_detections[0]  # Use first detected ball
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        self.ball_positions.append(center)
        
        # Check for dribble
        is_dribble = False
        if self.prev_center is not None:
            delta_y = center[1] - self.prev_center[1]
            if abs(delta_y) > self.dribble_threshold:
                is_dribble = True
                
        self.prev_center = center
        return center, is_dribble

class CourtTracker:
    def __init__(self):
        self.court_corners = None
        self.transform_matrix = None
        
    def set_court_corners(self, corners):
        """Set court corners for perspective transform"""
        if len(corners) != 4:
            raise ValueError("Need exactly 4 corner points")
            
        self.court_corners = np.array(corners)
        dst_points = np.array([[0, 0], [853, 0], [0, 640], [853, 640]])
        self.transform_matrix = cv2.getPerspectiveTransform(
            self.court_corners.astype(np.float32),
            dst_points.astype(np.float32)
        )
        
    def transform_coordinates(self, points):
        """Transform coordinates to court perspective"""
        if self.transform_matrix is None:
            return points
            
        transformed_points = cv2.perspectiveTransform(
            points.reshape(-1, 1, 2).astype(np.float32),
            self.transform_matrix
        )
        return transformed_points.reshape(-1, 2)