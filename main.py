import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import deque

from models.double_dribble import DoubleDribbleDetector
from models.shot_clock import ShotClockDetector
from models.travel import TravelDetector
from models.backcourt import BackcourtDetector
from models.ten_second import TenSecondDetector
from utils.visualization import ViolationVisualizer
from utils.tracking import PlayerTracker, BallTracker
from utils.input_handler import input_style
from config.settings import PROJECT_ROOT, MODEL_PATHS, VIDEO_SETTINGS, OUTPUT_DIRS

class BasketballRefereeSystem:
    def __init__(self, source_path='input/video/videoplayback.mp4'):
        self.source_path = source_path
        self.is_camera = source_path.startswith('camera:')
        self.is_video = not self.is_camera and source_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        
        # Initialize models and components
        self._initialize_models()
        self._initialize_components()
        
        # Setup video capture
        if self.is_camera:
            camera_id = int(source_path.split(':')[1])
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open camera {camera_id}")
        elif self.is_video:
            self.cap = cv2.VideoCapture(source_path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video: {source_path}")
            
        # Get frame dimensions for both video and camera
        if self.is_video or self.is_camera:
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._setup_video()
        
    def _initialize_models(self):
        # Initialize models first
        print("\nInitializing models...")
        self.models = {}
        
        try:
            self.models['pose'] = YOLO(MODEL_PATHS['pose'])
            print("✓ Pose model loaded successfully")
            
            self.models['ball'] = YOLO(MODEL_PATHS['ball'])
            print("✓ Ball model loaded successfully")
            
            self.models['player'] = YOLO(MODEL_PATHS['player'])
            print("✓ Player detection model loaded successfully")
            
            for model in self.models.values():
                model.overrides['imgsz'] = 1088
                
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
            
    def _initialize_components(self):
        # Pass models to detectors
        self.double_dribble = DoubleDribbleDetector(models=self.models)
        self.shot_clock = ShotClockDetector(models=self.models)
        self.travel = TravelDetector(models=self.models)
        self.backcourt = BackcourtDetector(models=self.models)
        self.ten_second = TenSecondDetector(models=self.models)
        
        # Initialize trackers
        self.player_tracker = PlayerTracker()
        self.ball_tracker = BallTracker()
        
        # Initialize frame buffer (reference from travel_detection.py)
        self.frame_buffer = deque(maxlen=VIDEO_SETTINGS['frame_buffer_size'])
        
        # Initialize visualization
        self.visualizer = ViolationVisualizer()
        
    def _setup_video(self):
        # Setup output video writer
        self.setup_video_writer()
        
    def setup_video_writer(self):
        Path(OUTPUT_DIRS['violation_clips']).mkdir(parents=True, exist_ok=True)
        
        # Get source name without extension
        if self.is_camera:
            source_name = f"camera_{self.source_path.split(':')[1]}"
        else:
            source_name = Path(self.source_path).stem
        
        output_path = str(Path(OUTPUT_DIRS['violation_clips']) / f"output_{source_name}.mp4")
        
        self.out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*VIDEO_SETTINGS['fourcc']),
            VIDEO_SETTINGS['fps'],
            (self.frame_width, self.frame_height)
        )
        
    def process_frame(self, frame):
        violations = []
        annotated_frame = frame.copy()
        
        # Ball detection (reference from travel_detection.py)
        ball_results_list = self.models['ball'](frame, verbose=False, conf=0.65)
        ball_detections = []
        
        for results in ball_results_list:
            for bbox in results.boxes.xyxy:
                ball_detections.append(bbox.cpu().numpy())
        
        # Pose detection (reference from double_dribble.py)
        pose_results = self.models['pose'](frame, verbose=False, conf=0.5)
        annotated_frame = pose_results[0].plot()
        
        # Player detection
        player_results = self.models['player'].predict(
            source=frame, 
            save=False, 
            imgsz=1080, 
            conf=0.15
        )
        player_detections = []
        
        for result in player_results:
            for box in result.boxes:
                if int(box.cls) == 1:  # Class 1 is for people
                    # Get bbox coordinates and confidence
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)
                    
                    # Convert to [x1,y1,x2,y2,conf] format
                    detection = np.append(xyxy, conf)
                    player_detections.append(detection)
        
        # Double dribble detection
        if self.double_dribble.check_violation(frame, ball_detections):
            violations.append("double_dribble")
        
        # Shot clock detection
        if self.shot_clock.check_violation(frame, player_detections):
            violations.append("shot_clock")
        
        # Travel detection
        if self.travel.check_violation(frame, pose_results, ball_results_list):
            violations.append("travel")
        
        # Backcourt detection
        if self.backcourt.check_violation(frame, ball_detections):
            violations.append("backcourt")
        
        # Ten second violation detection
        if self.ten_second.check_violation(frame, player_detections):
            violations.append("ten_second")
        
        # Update trackers
        player_tracks = self.player_tracker.update(player_detections, frame)
        ball_update = self.ball_tracker.update(ball_detections)
        ball_center = None
        is_dribble = False

        if ball_update is not None:
            ball_center, is_dribble = ball_update
        
        # Draw debug information
        self.draw_debug_info(annotated_frame, {
            "Players detected": len(player_detections),
            "Balls detected": len(ball_detections),
            "Tracking IDs": len(player_tracks)
        })
        
        return violations, annotated_frame
        
    def process_image(self):
        """Process single image and save output"""
        frame = cv2.imread(self.source_path)
        if frame is None:
            raise ValueError(f"Could not read image: {self.source_path}")
            
        violations, annotated_frame = self.process_frame(frame)
        
        # Visualize violations
        for violation in violations:
            annotated_frame = self.visualizer.add_violation_overlay(
                annotated_frame, 
                violation
            )
            
        # Save output
        output_path = Path(OUTPUT_DIRS['violation_clips']) / f"output_{Path(self.source_path).stem}.jpg"
        cv2.imwrite(str(output_path), annotated_frame)
        print(f"\nProcessed image saved to: {output_path}")
        
        # Display result
        cv2.imshow("Basketball Referee AI", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            
            if not success:
                break
                
            # Add frame to buffer
            self.frame_buffer.append(frame)
            
            # Process frame
            violations, annotated_frame = self.process_frame(frame)
            
            # Visualize violations
            for violation in violations:
                annotated_frame = self.visualizer.add_violation_overlay(
                    annotated_frame, 
                    violation
                )
            
            # Write frame
            self.out.write(annotated_frame)
            
            # Display frame
            cv2.imshow("Basketball Referee AI", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cleanup()
        
    def cleanup(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        
    def draw_debug_info(self, frame, info_dict):
        """Draw debug information on frame"""
        y_offset = 30
        for key, value in info_dict.items():
            cv2.putText(
                frame,
                f"{key}: {value}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            y_offset += 20



if __name__ == "__main__":
    source_path = input_style(PROJECT_ROOT)
    referee_system = BasketballRefereeSystem(source_path)
    
    try:
        if referee_system.is_video or referee_system.is_camera:
            referee_system.run()
        else:
            referee_system.process_image()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if referee_system.is_video or referee_system.is_camera:
            referee_system.cleanup()