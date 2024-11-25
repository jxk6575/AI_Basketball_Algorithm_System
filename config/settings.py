import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Model paths
MODEL_PATHS = {
    'pose': str(PROJECT_ROOT / "models" / "weights" / "yolov8s-pose.pt"),
    'ball': str(PROJECT_ROOT / "models" / "weights" / "basketballModel.pt"),
    'player': str(PROJECT_ROOT / "models" / "weights" / "best_1_27.pt")
}

# Video settings
VIDEO_SETTINGS = {
    'frame_width': 1920,
    'frame_height': 1080,
    'fps': 30,
    'fourcc': 'mp4v',
    'output_dir': 'output',
    'frame_buffer_size': 30,
    'model_imgsz': 1088
}

# Detection thresholds
DETECTION_THRESHOLDS = {
    'ball_confidence': 0.5,
    'pose_confidence': 0.5,
    'distance': 50,
    'hold_duration': 0.85,
    'hold_threshold': 300,
    'dribble_threshold': 18,
    'step_threshold': 5,
    'interaction_distance': 50
}

# Violation settings
VIOLATION_SETTINGS = {
    'shot_clock': 24,
    'backcourt': 8,
    'max_ball_not_detected_frames': 20
}

# Player tracking settings (ReID)
TRACKING_SETTINGS = {
    'reid_enabled': True,
    'appearance_thresh': 0.5,
    'proximity_thresh': 0.5,
    'track_buffer': 30
}

# Body keypoint indices
BODY_INDICES = {
    'left_wrist': 10,
    'right_wrist': 9,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Output directories
OUTPUT_DIRS = {
    'travel_footage': str(PROJECT_ROOT / "output"),
    'violation_clips': str(PROJECT_ROOT / "output")
}

# Create required directories
for directory in OUTPUT_DIRS.values():
    os.makedirs(directory, exist_ok=True)

# Model settings
MODEL_SETTINGS = {
    'imgsz': 1088,  # Multiple of 32 for YOLO models
    'conf_threshold': {
        'pose': 0.5,
        'ball': 0.65,
        'player': 0.7
    }
}

# Visualization settings
VISUALIZATION_SETTINGS = {
    'show_reid': True,
    'show_track_history': True,
    'track_history_length': 30,
    'reid_text_color': (255, 255, 255),  # White text
    'reid_box_thickness': 2,
    'reid_text_scale': 0.5,
}