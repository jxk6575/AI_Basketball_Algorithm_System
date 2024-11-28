# Basketball Game Algorithm Implementation Details

## Project Overview
This project implements an AI-based basketball referee system that automatically detects various violations in basketball games using computer vision and deep learning techniques. The system processes video feeds in real-time or from stored footage, utilizing multiple specialized neural networks for different aspects of game analysis.

## Core Components

### 1. Violation Detection Models
Located in `models/` directory:

#### Base Detector (`detector_base.py`)
- Abstract base class `BaseDetector`
- Implements common detection pipeline:
  - Model loading and initialization
  - Frame preprocessing
  - Result post-processing
  - Debug visualization
- Memory management for efficient processing
- Error handling and recovery mechanisms

#### Specific Violation Detectors
- **Blocking Foul Detector**
  - Uses YOLOv8-pose for skeletal tracking
  - Implements collision detection algorithm
  - Analyzes player stance and momentum vectors
  - Confidence threshold: 0.5 for pose detection
  
- **Double Dribble Detector**
  - Combines ball tracking and player pose estimation
  - State machine for dribble sequence analysis
  - Temporal analysis window: 30 frames
  - Uses ball possession heuristics
  
- **Shot Clock Detector**
  - Frame-accurate possession timing
  - Team possession tracking using player positions
  - Clock reset detection on specific events
  - Configurable violation threshold (24s default)
  
- **Ten Second Detector**
  - Court region segmentation
  - Player position tracking in backcourt
  - Continuous timing system
  - Handles interrupted possessions
  
- **Backcourt Detector**
  - Court line detection using computer vision
  - Player tracking across court regions
  - Team possession state management
  - Violation detection with spatial constraints
  
- **Travel Detector**
  - Foot movement analysis using pose estimation
  - Step counter with temporal filtering
  - Ball possession correlation
  - Pivot foot detection algorithm

### 2. Tracking System
Located in `utils/tracking.py`:

#### Player Tracking
- **BYTE Tracker Integration**
  - Motion prediction
  - Track association using IoU and appearance
  - Track management (creation/deletion)
  - Occlusion handling
  
- **ReID System**
  - Feature extraction using FastReID
  - Feature matching with cosine similarity
  - Temporal feature buffer (10 frames)
  - Re-identification confidence threshold: 0.5

#### Motion Analysis
- Kalman filtering for smooth trajectories
- Velocity and acceleration computation
- Direction vector analysis
- Collision prediction

### 3. Input Processing System
Located in `utils/input_handler.py`:

#### Video Processing
- Multi-format support (MP4, AVI, MOV)
- Frame extraction and buffering
- Resolution standardization
- Frame rate management

#### Real-time Processing
- Camera input handling
- Frame synchronization
- Buffer management
- Latency optimization

### 4. Configuration Management
Located in `config/settings.py`:

#### Model Settings
```python
MODEL_SETTINGS = {
    'imgsz': 1088,  # Input resolution
    'conf_threshold': {
        'pose': 0.5,
        'ball': 0.65,
        'player': 0.7
    },
    'track_args': {
        'track_thresh': 0.15,
        'track_buffer': 30,
        'match_thresh': 0.7
    }
}
```

#### Detection Parameters
```python
DETECTION_THRESHOLDS = {
    'ball_confidence': 0.5,
    'pose_confidence': 0.5,
    'distance': 50,
    'hold_duration': 0.85
}
```

## Technical Implementation Details

### 1. Deep Learning Pipeline
- **Model Architecture**
  - YOLOv8-pose for human pose estimation
  - Custom-trained YOLOv5 for ball detection
  - FastReID for player identification
  
- **Inference Optimization**
  - CUDA acceleration
  - Batch processing
  - Model quantization
  - TensorRT integration (optional)

### 2. Computer Vision Pipeline
- **Pre-processing**
  - Frame resizing (1088x1088)
  - Normalization
  - Color space conversion
  - Perspective correction
  
- **Post-processing**
  - Non-maximum suppression
  - Temporal smoothing
  - Confidence filtering
  - Coordinate transformation

### 3. Data Flow
1. Input Source → Frame Buffer
   - Frame extraction
   - Resolution standardization
   - Color space conversion

2. Frame Buffer → Detection Models
   - Batch formation
   - GPU transfer
   - Parallel inference

3. Detection Results → Violation Analysis
   - Result aggregation
   - Temporal correlation
   - Rule violation checking

4. Violation Analysis → Output Generation
   - Event logging
   - Visual annotation
   - Alert generation
   - Video encoding

### 4. Performance Optimization
- **Memory Management**
  - Frame buffer size: 30 frames
  - Feature buffer size: 10 frames
  - GPU memory optimization
  - Batch size adaptation

- **Real-time Processing**
  - Multi-threading for I/O
  - Parallel inference
  - Asynchronous video writing
  - Frame skipping when necessary

### 5. Error Handling
- Model failure recovery
- Input stream interruption handling
- GPU memory management
- Exception logging and reporting

## Development Environment

### Hardware Requirements
- CPU: Intel i5/AMD Ryzen 5 or better
- RAM: 16GB minimum
- GPU: NVIDIA GPU with 6GB+ VRAM
- Storage: 5GB for models and cache

### Software Dependencies
- Python 3.12
- PyTorch 2.4.1
- OpenCV 4.8.0
- CUDA 11.8+
- cuDNN 8.7+

### Development Tools
- VSCode/PyCharm with Python support
- Git for version control
- Docker for deployment
- NVIDIA NSight for profiling

## Testing and Validation
- Unit tests for components
- Integration testing
- Performance benchmarking
- Accuracy validation against human referees
- System stress testing

## Future Improvements
1. Additional violation types
2. Multi-camera support
3. 3D pose estimation
4. Advanced team tactics analysis
5. Automated highlight generation
6. Cloud deployment support

## Model Weights and Architecture Details

### 1. Model Files Overview
Located in `models/weights/` directory:

#### YOLOv8-Pose Model
**File**: `yolov8s-pose.pt`
- **Size**: ~42MB
- **Purpose**: Human pose estimation
- **Architecture**: YOLOv8-small with pose estimation head
- **Input size**: 1088x1088 pixels
- **Output**: 17 keypoints per person
- **Used by**:
  - BlockingFoulDetector (player stance analysis)
  - TravelDetector (foot movement tracking)
  - DoubleDribbleDetector (player movement analysis)
- **Download**: Available from Ultralytics official repository
- **Inference speed**: ~25ms per frame on RTX 3060

#### Basketball Detection Model
**File**: `basketballModel.pt`
- **Size**: ~15MB
- **Purpose**: Basketball detection and tracking
- **Architecture**: Custom YOLOv5-small
- **Input size**: 640x640 pixels
- **Output**: Ball bounding boxes and confidence scores
- **Used by**:
  - DoubleDribbleDetector (ball possession)
  - TravelDetector (ball relationship)
- **Training**: Custom trained on basketball dataset
  - 10,000 annotated frames
  - Augmentation: rotation, scaling, brightness
  - Training time: 100 epochs
- **Inference speed**: ~15ms per frame on RTX 3060

#### Player Detection Model
**File**: `best_1_27.pt`
- **Size**: ~28MB
- **Purpose**: Player detection and tracking
- **Architecture**: YOLOv5-medium with custom head
- **Input size**: 1088x1088 pixels
- **Output**: Player bounding boxes and team classification
- **Used by**:
  - ShotClockDetector (possession tracking)
  - BackcourtDetector (court position)
  - TenSecondDetector (backcourt timing)
- **Training**: Fine-tuned on basketball game footage
  - 25,000 frames from professional games
  - Team classification head added
  - Custom anchor optimization
- **Inference speed**: ~20ms per frame on RTX 3060

### 2. Model Loading and Management

#### Loading Mechanism
```python
# Located in detector_base.py
def _load_models(self, model_paths):
    for name, path in model_paths.items():
        model = YOLO(str(path))
        model.overrides['imgsz'] = MODEL_SETTINGS['imgsz']
        self.models[name] = model
```

#### Memory Management
- Dynamic model loading/unloading
- CUDA memory optimization
- Batch size adaptation
- Model quantization options

#### Inference Optimization
- TensorRT export support
- Half-precision (FP16) inference
- Batch processing
- CUDA stream management

### 3. Model Version Compatibility

#### YOLOv8-Pose
- Compatible versions: 8.0.0+
- CUDA requirement: 11.4+
- TensorRT support: 8.4+
- OpenCV requirement: 4.6+

#### Custom Models
- YOLOv5 compatibility: 6.0+
- PyTorch versions: 1.8+
- Torchvision: 0.9+
- ONNX support: 1.12+

### 4. Model Deployment

#### Weight File Management
- Automatic download if missing
- Checksum verification
- Version control integration
- Backup management

#### Conversion Options
- ONNX export support
- TensorRT optimization
- Int8 quantization
- Mobile deployment formats

### 5. Performance Considerations

#### GPU Memory Usage
- YOLOv8-Pose: ~2.1GB
- Basketball Model: ~0.8GB
- Player Detection: ~1.4GB
- Total peak memory: ~4.5GB

#### Batch Processing
- Optimal batch size: 4
- Memory scaling: ~1.6x per batch
- Latency vs throughput tradeoff
- Dynamic batch adaptation

#### Multi-GPU Support
- Model parallelization
- Device management
- Load balancing
- Memory optimization

### 6. Custom Training Guidelines

#### Data Preparation
- Annotation format: YOLO format
- Resolution requirements
- Augmentation pipeline
- Dataset organization

#### Training Process
- Hyperparameter optimization
- Loss function modifications
- Validation metrics
- Early stopping criteria

#### Model Export
- Format conversion
- Optimization steps
- Deployment preparation
- Testing procedures
