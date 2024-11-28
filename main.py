import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import deque
import torch

from models.double_dribble import DoubleDribbleDetector
from models.shot_clock import ShotClockDetector
from models.travel import TravelDetector
from models.backcourt import BackcourtDetector
from models.blocking_foul import BlockingFoulDetector
from models.ten_second import TenSecondDetector
from utils.visualization import ViolationVisualizer
from utils.tracking import PlayerTracker, BYTETracker
from utils.input_handler import input_style
from config.settings import PROJECT_ROOT, MODEL_PATHS, VIDEO_SETTINGS, OUTPUT_DIRS, MODEL_SETTINGS


class TrackerArgs:
    """Arguments for BYTETracker initialization"""
    def __init__(self):
        track_args = MODEL_SETTINGS['track_args']
        self.track_thresh = track_args['track_thresh']
        self.track_buffer = track_args['track_buffer']
        self.match_thresh = track_args['match_thresh']
        self.mot20 = track_args['mot20']
        self.track_high_thresh = track_args['track_high_thresh']
        self.track_low_thresh = track_args['track_low_thresh']
        self.new_track_thresh = track_args['new_track_thresh']
        self.frame_rate = track_args['frame_rate']
        self.aspect_ratio_thresh = track_args['aspect_ratio_thresh']
        self.min_box_area = track_args['min_box_area']


class BasketballRefereeSystem:
    def __init__(self, source_path):
        self.source_path = source_path
        self.is_camera = source_path.startswith('camera:')
        self.is_video = not self.is_camera and Path(source_path).suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv')
        self.is_image = not (self.is_camera or self.is_video)
        self.save_output = True
        self.current_violations = []

        # Initialize models and components
        self._initialize_models()
        self._initialize_components()
        self._setup_source()

    def _setup_source(self):
        """Setup input source and output paths"""
        if self.is_camera:
            camera_id = int(self.source_path.split(':')[1])
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open camera {camera_id}")
            self.output_name = f"camera_{camera_id}"
        
        elif self.is_video:
            self.cap = cv2.VideoCapture(self.source_path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video: {self.source_path}")
            self.output_name = Path(self.source_path).stem
        
        else:  # Image
            self.frame = cv2.imread(self.source_path)
            if self.frame is None:
                raise ValueError(f"Could not read image: {self.source_path}")
            self.output_name = Path(self.source_path).stem

        # Create output directories if they don't exist
        for dir_path in OUTPUT_DIRS.values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Setup video writer if needed
        if (self.is_camera or self.is_video) and self.save_output:
            self._setup_video_writer()

    def _setup_video_writer(self):
        """Initialize video writer with proper settings"""
        if not hasattr(self, 'cap'):
            return

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) if self.is_video else 30
        
        # Add fixed info panel height if visualizer isn't initialized yet
        info_panel_height = getattr(self.visualizer, 'info_panel_height', 100)
        total_height = height + info_panel_height

        output_path = str(Path(OUTPUT_DIRS['violation_clips']) / f"{self.output_name}_output.mp4")
        self.out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*VIDEO_SETTINGS['fourcc']),
            fps,
            (width, total_height)
        )
        print(f"\nSaving output to: {output_path}")

    def _initialize_models(self):
        """Initialize all required models"""
        print("\nInitializing models...")
        try:
            self.models = {
                'pose': YOLO(MODEL_PATHS['pose']),
                'ball': YOLO(MODEL_PATHS['ball']),
                'player': YOLO(MODEL_PATHS['player'])
            }
            
            # Set image size for all models
            for model in self.models.values():
                model.overrides['imgsz'] = MODEL_SETTINGS['imgsz']
                
        except Exception as e:
            print(f"Error initializing models: {e}")
            raise

    def _initialize_components(self):
        """Initialize all components"""
        # Initialize detectors with shared models
        self.double_dribble = DoubleDribbleDetector(self.models)
        self.shot_clock = ShotClockDetector(self.models)
        self.travel = TravelDetector(self.models)
        self.backcourt = BackcourtDetector(self.models)
        self.blocking_foul = BlockingFoulDetector(self.models)
        self.ten_second = TenSecondDetector(self.models)

        # Initialize trackers with proper configuration
        tracker_args = TrackerArgs()
        self.player_tracker = PlayerTracker()
        self.ball_tracker = BYTETracker(tracker_args)

        # Initialize visualizer
        self.visualizer = ViolationVisualizer()

    def process_frame(self, frame):
        violations = []
        
        # Ball detection with proper formatting
        ball_results = self.models['ball'](frame, verbose=False, conf=0.65)
        ball_detections = []
        if len(ball_results) > 0:
            boxes = ball_results[0].boxes
            if len(boxes) > 0:
                # Combine boxes and confidence scores
                ball_detections = np.column_stack([
                    boxes.xyxy.cpu().numpy(),
                    boxes.conf.cpu().numpy()
                ])

        # Player detection with proper formatting
        player_results = self.models['player'](
            frame,
            conf=0.2,
            iou=0.45,
            verbose=False
        )
        
        player_detections = []
        if len(player_results) > 0:
            boxes = player_results[0].boxes
            if len(boxes) > 0:
                # Combine boxes and confidence scores
                player_detections = np.column_stack([
                    boxes.xyxy.cpu().numpy(),
                    boxes.conf.cpu().numpy()
                ])

        # Pose detection
        pose_results = self.models['pose'](frame, verbose=False, conf=0.5)
        annotated_frame = pose_results[0].plot()

        # Update trackers with properly formatted detections
        tracked_players = self.player_tracker.update(player_detections, frame)
        
        # Create Results object for ball tracker
        class Results:
            def __init__(self, dets):
                self.boxes = torch.from_numpy(dets[:, :4]) if len(dets) > 0 else torch.empty((0, 4))
                self.conf = torch.from_numpy(dets[:, 4]) if len(dets) > 0 else torch.empty(0)
                self.cls = torch.zeros(len(dets))
                self.xyxy = self.boxes

        ball_results_obj = Results(ball_detections)
        try:
            ball_tracks = self.ball_tracker.update(
                ball_results_obj,
                [frame.shape[0], frame.shape[1]]
            )
        except Exception as e:
            print(f"Ball tracker error: {str(e)}")
            ball_tracks = []

        # Check violations
        if self.double_dribble.check_violation(frame, ball_detections):
            violations.append("Double Dribble")
        if self.shot_clock.check_violation(frame, player_detections):
            violations.append("Shot Clock")
        if self.travel.check_violation(frame, pose_results, ball_results):
            violations.append("Travel")
        if self.backcourt.check_violation(frame, ball_detections):
            violations.append("Backcourt")
        if self.blocking_foul.check_violation(frame, ball_detections):
            violations.append("Blocking Foul")
        if self.ten_second.check_violation(frame, player_detections):
            violations.append("Ten Second")

        # Create final visualization (only once!)
        final_frame = self.visualizer.visualize_frame(
            annotated_frame,
            tracked_players,
            violations
        )

        return violations, final_frame, tracked_players

    def process_image(self):
        """Process single image"""
        violations, annotated_frame, tracked_players = self.process_frame(self.frame)
        
        # Create final visualization
        final_frame = self.visualizer.visualize_frame(
            annotated_frame,
            tracked_players,
            violations
        )

        # Save output
        output_path = Path(OUTPUT_DIRS['violation_clips']) / f"{self.output_name}_output.jpg"
        cv2.imwrite(str(output_path), final_frame)
        print(f"\nSaved output to: {output_path}")

        # Display result
        cv2.imshow("Basketball Referee System", final_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run(self):
        """Main processing loop for video/camera input"""
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            print("Error: Video capture not initialized")
            return

        try:
            frame_count = 0
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame_count += 1
                print(f"\rProcessing frame: {frame_count}", end="")

                # Process frame
                violations, final_frame, tracked_players = self.process_frame(frame)
                
                # Display frame (don't add info panel again!)
                cv2.imshow('Basketball Referee System', final_frame)

                if self.save_output and self.out is not None:
                    self.out.write(final_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"\nError during processing: {e}")

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'out') and self.out is not None:
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
    try:
        source_path = input_style(PROJECT_ROOT)
        referee_system = BasketballRefereeSystem(source_path)
        
        if referee_system.is_image:
            referee_system.process_image()
        else:
            referee_system.run()
            
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()