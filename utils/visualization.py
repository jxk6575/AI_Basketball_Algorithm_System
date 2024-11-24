import cv2
import numpy as np

class ViolationVisualizer:
    COLORS = {
        'double_dribble': (0, 0, 255),  # Red
        'shot_clock': (255, 0, 0),      # Blue
        'blocking_foul': (0, 255, 0)     # Green
    }
    
    @staticmethod
    def add_violation_overlay(frame, violation_type, duration=3):
        color = ViolationVisualizer.COLORS.get(violation_type, (0, 0, 255))
        tint = np.full_like(frame, color, dtype=np.uint8)
        overlay = cv2.addWeighted(frame, 0.7, tint, 0.3, 0)
        
        cv2.putText(
            overlay,
            f"{violation_type.replace('_', ' ').title()}!",
            (frame.shape[1] - 600, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )
        return overlay