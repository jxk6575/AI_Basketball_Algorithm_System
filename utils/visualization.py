import cv2
import numpy as np

class ViolationVisualizer:
    def __init__(self):
        self.colors = {}  # Store unique colors for each track ID
        self.info_panel_height = 100  # Height of bottom info panel in pixels
        
    def get_color(self, track_id):
        if track_id not in self.colors:
            # Generate unique color for this ID
            self.colors[track_id] = (
                int((track_id * 47) % 255),
                int((track_id * 97) % 255),
                int((track_id * 157) % 255)
            )
        return self.colors[track_id]

    def draw_reid_info(self, frame, tracked_players):
        """Draw ReID information and tracked players"""
        # Create a copy of the frame
        annotated_frame = frame.copy()
        
        # Draw tracked players
        for player in tracked_players:
            # Get track ID and box
            track_id = player.track_id
            box = player.bbox.astype(int)
            color = self.get_color(track_id)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, 
                         (box[0], box[1]), 
                         (box[2], box[3]), 
                         color, 2)
            
            # Draw ID text with background
            text = f"ID: {track_id}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Draw text background
            cv2.rectangle(annotated_frame,
                         (box[0], box[1] - text_size[1] - 8),
                         (box[0] + text_size[0], box[1]),
                         color, -1)
            
            # Draw text
            cv2.putText(annotated_frame, text,
                       (box[0], box[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame

    def add_info_panel(self, frame, tracked_players, violations=None):
        """Add information panel at the bottom of the frame"""
        h, w = frame.shape[:2]
        
        # Create white panel
        panel = np.ones((self.info_panel_height, w, 3), dtype=np.uint8) * 255
        
        # Add tracking information
        text = f"Tracked Players: {len(tracked_players)}"
        cv2.putText(panel, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add violations if any
        if violations:
            violation_text = f"Violations: {', '.join(violations)}"
            cv2.putText(panel, violation_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Combine frame and panel
        result = np.vstack([frame, panel])
        
        return result

    def visualize_frame(self, frame, tracked_players, violations=None):
        """Complete visualization pipeline"""
        # Draw ReID information
        frame_with_reid = self.draw_reid_info(frame, tracked_players)
        
        # Add information panel
        final_frame = self.add_info_panel(frame_with_reid, tracked_players, violations)
        
        return final_frame