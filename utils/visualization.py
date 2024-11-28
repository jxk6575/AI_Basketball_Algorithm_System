import cv2
import numpy as np

class ViolationVisualizer:
    def __init__(self):
        self.colors = {}  # Store unique colors for each track ID
        self.info_panel_height = 100  # Height of bottom info panel in pixels
        
    def get_color(self, track_id):
        """Generate unique color for each track ID using HSV color space"""
        if track_id not in self.colors:
            # Use golden ratio to generate well-distributed hues
            hue = (track_id * 0.618033988749895) % 1.0
            # Convert HSV to RGB (using full saturation and value)
            rgb = cv2.cvtColor(
                np.uint8([[[hue * 255, 255, 255]]]),
                cv2.COLOR_HSV2BGR
            )[0][0]
            self.colors[track_id] = tuple(map(int, rgb))
        return self.colors[track_id]

    def draw_reid_info(self, frame, tracked_players):
        """Draw ReID information on frame with unique colors"""
        for player in tracked_players:
            box = player.bbox.astype(int)
            
            # Get unique color for this ID
            color = self.get_color(player.track_id)
            
            # Draw bounding box with unique color
            cv2.rectangle(frame, 
                         (box[0], box[1]), 
                         (box[2], box[3]), 
                         color, 2)
            
            # Draw ID with background for better visibility
            text = f'ID: {player.track_id}'
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw text background
            cv2.rectangle(frame,
                         (box[0], box[1] - text_height - 8),
                         (box[0] + text_width + 4, box[1]),
                         color, -1)  # -1 fills the rectangle
            
            # Draw text in white for contrast
            cv2.putText(frame, text,
                       (box[0] + 2, box[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame

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
        # Draw ReID information first
        frame_with_reid = self.draw_reid_info(frame.copy(), tracked_players)
        
        # Add information panel once
        final_frame = self.add_info_panel(frame_with_reid, tracked_players, violations)
        
        return final_frame