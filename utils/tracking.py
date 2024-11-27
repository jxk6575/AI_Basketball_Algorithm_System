from ultralytics.trackers.byte_tracker import BYTETracker, STrack
from collections import deque
import numpy as np
import torch
import cv2
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


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
        if len(detections) == 0:
            return []

        try:
            # Apply NMS to remove duplicate detections
            from torchvision.ops import nms
            boxes = torch.from_numpy(detections[:, :4])
            scores = torch.from_numpy(detections[:, 4])
            keep_indices = nms(boxes, scores, iou_threshold=0.45)
            detections = detections[keep_indices.cpu().numpy()]

            # Create Results object for tracker
            class Results:
                def __init__(self, dets):
                    self.boxes = torch.from_numpy(dets[:, :4])
                    self.conf = torch.from_numpy(dets[:, 4])
                    self.cls = torch.zeros(len(dets))
                    self.xyxy = self.boxes

            results = Results(detections)

            # Get tracker output
            online_targets = self.tracker.update(
                results,
                [frame.shape[0], frame.shape[1]]
            )

            # Match current boxes with previous frame's boxes
            current_boxes = [target[:4] for target in online_targets]
            matched_ids = self.match_boxes(current_boxes)

            # Update track ages
            current_track_ids = set(matched_ids.values())
            for track_id in list(self.track_ages.keys()):
                if track_id in current_track_ids:
                    self.track_ages[track_id] = 0
                else:
                    self.track_ages[track_id] += 1
                    if self.track_ages[track_id] > self.max_age:
                        del self.track_ages[track_id]
                        if track_id in self.track_history:
                            del self.track_history[track_id]

            # Convert to tracked objects
            tracked_objects = []
            for i, target in enumerate(online_targets):
                if isinstance(target, np.ndarray) and len(target) >= 4:
                    track_id = matched_ids[i]

                    class TrackedObject:
                        def __init__(self, box, id_, score=None):
                            self.track_id = id_
                            self.bbox = box[:4]
                            self.score = float(score) if score is not None else 1.0
                            self.cls = 0
                            self.tlbr = box[:4]

                    score = target[4] if len(target) > 4 else None
                    tracked_obj = TrackedObject(target, track_id, score)

                    # Update track history
                    center = ((int(target[0]) + int(target[2])) // 2,
                              (int(target[1]) + int(target[3])) // 2)

                    if track_id not in self.track_history:
                        self.track_history[track_id] = deque(maxlen=30)

                    self.track_history[track_id].append(center)
                    tracked_obj.track_history = list(self.track_history[track_id])

                    tracked_objects.append(tracked_obj)

                    # Update last known position
                    self.last_boxes[track_id] = target[:4]
                    self.track_ages[track_id] = 0

            # Clean up old tracks
            self.last_boxes = {k: v for k, v in self.last_boxes.items()
                               if k in self.track_ages}

            # Extract features for each detection
            for det in detections:
                if det is not None:
                    bbox = det[:4]
                    conf = det[4]
                    
                    # Extract image patch
                    x1, y1, x2, y2 = map(int, bbox)
                    patch = frame[y1:y2, x1:x2]
                    
                    # Update feature buffer
                    if patch.size > 0:  # Ensure valid patch
                        feature = self.extract_features(patch)
                        self.update_features(det[-1], feature)
                        
            return tracked_objects

        except Exception as e:
            print(f"Error in tracker update: {str(e)}")
            print(f"Error details: {type(e).__name__}")
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
