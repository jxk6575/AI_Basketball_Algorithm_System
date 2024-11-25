# Basketball Game Algorithm Implementation Details

## Project Overview
This project implements an AI-based basketball referee system that can automatically detect various violations in basketball games using computer vision and deep learning techniques.

## Core Components

### 1. Violation Detection Models
Located in `models/` directory:

#### Base Detector
- Abstract base class `BaseDetector` in `detector_base.py`
- Defines common interface for all violation detectors
- Handles basic detection pipeline

#### Specific Violation Detectors
- **Blocking Foul Detector**
  - Tracks player positions and detects illegal blocking
  - Uses Euclidean distance calculations for player proximity
  
- **Double Dribble Detector**
  - Monitors ball possession and dribbling sequences
  - Detects illegal second dribble after stopping
  
- **Shot Clock Detector**
  - Tracks possession time
  - Monitors 24-second violation
  
- **Ten Second Detector**
  - Tracks backcourt time
  - Enforces 10-second rule
  
- **Backcourt Detector**
  - Monitors court boundaries
  - Detects illegal returns to backcourt
  
- **Travel Detector**
  - Analyzes player foot movements
  - Detects illegal steps without dribbling

### 2. Utility Functions
Located in `utils/` directory:

#### Tracking (`tracking.py`)
- Player tracking using YOLOv5 + DeepSORT
- Player re-identification using FastReID
- Motion tracking and trajectory analysis

#### Visualization (`visualization.py`)
- Real-time annotation of detected violations
- Visual debugging tools
- Results visualization

#### Input Processing (`input_handler.py`)
- Video/image input processing
- Frame extraction and preprocessing
- Data format standardization

### 3. Configuration Management
Located in `config/` directory:

#### Settings (`settings.py`)
- Detection parameters
- Model configurations
- System settings
- Path configurations

## Technical Implementation Details

### Player Detection and Tracking
1. YOLOv5 for initial player detection
2. DeepSORT for tracking continuity
3. FastReID for player re-identification
4. Custom tracking refinements for basketball context

### Violation Detection Algorithms
1. **Blocking Foul**
   - Position tracking
   - Velocity vectors
   - Contact detection
   
2. **Double Dribble**
   - Ball possession tracking
   - Dribble sequence analysis
   - State machine implementation
   
3. **Shot Clock**
   - Possession timer
   - Team possession tracking
   - Clock reset detection

### Data Flow
1. Input video/image processing
2. Player detection and tracking
3. Feature extraction
4. Violation detection
5. Result visualization
6. Output generation

## Development Environment

### Requirements
- Python 3.12
- PyTorch
- OpenCV
- NumPy
- YOLOv5
- DeepSORT
- FastReID

### Development Tools
- PyCharm IDE
- Git version control
- Remote deployment support

## Testing and Validation
- Unit tests for individual components
- Integration testing for full pipeline
- Performance benchmarking
- Accuracy validation

## Future Improvements
1. Additional violation types
2. Performance optimization
3. Real-time processing
4. Multi-camera support
5. Enhanced visualization
