from ultralytics.trackers.byte_tracker import BYTETracker, STrack
from collections import deque
import numpy as np
import torch
import cv2
import warnings
from torchvision.ops import nms
warnings.filterwarnings('ignore', category=FutureWarning)
from .reid_module import ReIDModule
from pathlib import Path
from filterpy.kalman import KalmanFilter


class DetectionResults:
    """Class to format detections for BYTETracker"""
    def __init__(self, dets):
        self.boxes = torch.from_numpy(dets[:, :4]) if len(dets) > 0 else torch.empty((0, 4))
        self.conf = torch.from_numpy(dets[:, 4]) if len(dets) > 0 else torch.empty(0)
        self.cls = torch.zeros(len(dets))
        self.xyxy = self.boxes

class Track:
    """Class to store track information"""
    def __init__(self, tlbr, track_id, score=1.0):
        self.tlbr = tlbr
        self.track_id = track_id
        self.score = score
        self.bbox = np.array(tlbr, dtype=np.float32)
        self.reid_features = None


class PlayerTracker:
    def __init__(self, track_buffer=30):
        self.tracks = {}  # {track_id: last_box}
        self.track_history = {}
        self.next_id = 0
        self.last_boxes = {}  # Store last frame's boxes
        self.track_ages = {}  # Track how long since we've seen each ID
        self.max_age = 30  # Maximum frames to keep a track alive

        # Initialize BYTETracker
        class Args:
            track_thresh = 0.15
            track_buffer = 30
            match_thresh = 0.7
            mot20 = False
            track_high_thresh = 0.3
            track_low_thresh = 0.05
            new_track_thresh = 0.2
            frame_rate = 30
            aspect_ratio_thresh = 1.6
            min_box_area = 50

        self.tracker = BYTETracker(Args())

        # Add appearance feature buffer
        self.feature_buffer = {}  # {track_id: deque of features}
        self.feature_buffer_size = 10

        # Add safe loading configuration
        torch.serialization.add_safe_globals(
            {'torch': torch, 'np': np}
        )

        # Initialize ReID module
        self.reid_module = ReIDModule(
            model_path=str(Path(__file__).parent.parent / "models" / "weights" / "osnet_x0_25.pth"),
            threshold=0.5
        )

    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)

    def match_boxes(self, current_boxes):
        """Match current boxes with previous frame's boxes"""
        if not self.last_boxes:
            return {i: self.next_id + i for i in range(len(current_boxes))}

        # Calculate IoU matrix
        iou_matrix = np.zeros((len(current_boxes), len(self.last_boxes)))
        for i, curr_box in enumerate(current_boxes):
            for j, (track_id, last_box) in enumerate(self.last_boxes.items()):
                iou_matrix[i, j] = self.calculate_iou(curr_box, last_box)

        # Match boxes based on IoU
        matched_ids = {}
        iou_threshold = 0.3

        # First, handle high-confidence matches
        while True:
            if not iou_matrix.size:
                break
            i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            if iou_matrix[i, j] < iou_threshold:
                break

            track_id = list(self.last_boxes.keys())[j]
            matched_ids[i] = track_id
            iou_matrix[i, :] = -1
            iou_matrix[:, j] = -1

        # Assign new IDs to unmatched boxes
        for i in range(len(current_boxes)):
            if i not in matched_ids:
                matched_ids[i] = self.next_id
                self.next_id += 1

        return matched_ids

    def update_features(self, track_id, new_feature):
        """Update appearance features for ReID"""
        if track_id not in self.feature_buffer:
            self.feature_buffer[track_id] = deque(maxlen=self.feature_buffer_size)
        self.feature_buffer[track_id].append(new_feature)
        
    def compute_similarity(self, feature1, feature2):
        """Compute cosine similarity between features"""
        return np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
        
    def update(self, detections, frame):
        """Update tracks with new detections"""
        try:
            detections_array = np.array(detections)
            
            if len(detections_array.shape) == 2 and detections_array.shape[1] >= 5:
                results = DetectionResults(detections_array)
                
                tracks = self.tracker.update(
                    results,
                    [frame.shape[0], frame.shape[1]]
                )
                
                tracked_objects = []
                for track in tracks:
                    # BYTETracker returns numpy array: [x1, y1, x2, y2, score, class_id]
                    x1, y1, x2, y2 = track[:4]  # First 4 elements are coordinates
                    track_id = int(track[4])  # 5th element is track_id
                    score = float(track[5]) if len(track) > 5 else 1.0  # 6th element is score if available
                    
                    track_obj = Track([x1, y1, x2, y2], track_id, score)
                    tracked_objects.append(track_obj)
                
                return tracked_objects
            else:
                print(f"Invalid detection format. Expected shape (N, 5+), got {detections_array.shape}")
                return []

        except Exception as e:
            print(f"Error in tracker update: {str(e)}")
            print(f"Error details: {type(e).__name__}")
            print(f"Detection type when error occurred: {type(detections)}")
            import traceback
            traceback.print_exc()
            return []

    def extract_features(self, patch):
        """Extract appearance features from image patch"""
        # Ensure patch is valid
        if patch.size == 0:
            return np.zeros(512)  # Return zero feature vector
            
        # Resize patch to standard size
        patch = cv2.resize(patch, (64, 128))
        
        # Convert to float32 and normalize
        patch = patch.astype(np.float32) / 255.0
        
        # Simple feature extraction (can be enhanced with a proper ReID model)
        feature = cv2.mean(patch)[:3]  # Use mean color as feature
        return np.array(feature)
