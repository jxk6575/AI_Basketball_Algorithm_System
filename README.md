# Basketball Game Algorithm

An AI-powered basketball referee system that automatically detects rule violations using computer vision and deep learning techniques. The system analyzes basketball game footage to identify various violations including blocking fouls, double dribble, shot clock violations, and more.
<img width="472" alt="7f50e102e6966faa7f5366020598a55" src="https://github.com/user-attachments/assets/9605b022-2cdd-42bc-b873-47801b670801">


## Features

### Violation Detection
- **Blocking Fouls**: Detects illegal blocking positions and player contact
- **Double Dribble**: Identifies illegal second dribbles
- **Shot Clock**: Monitors 24-second possession violations
- **Ten Second**: Tracks backcourt violations
- **Backcourt**: Detects illegal returns to backcourt
- **Traveling**: Analyzes player movements for illegal steps

### Technical Capabilities
- Real-time player tracking using YOLOv5
- Multi-object tracking with DeepSORT
- Pose estimation for detailed movement analysis
- Video processing and annotation
- Cloud deployment support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jxk6575/AI_Basketball_Algorithm_System.git
cd basketball_game_algorithm
```

2. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download model weights:
```bash
mkdir -p models/weights
# Download the following weights to models/weights/:
# - yolov8s-pose.pt
# - basketballModel.pt
# - best_1_27.pt
```

## Project Structure
```
basketball_game_algorithm/
├── config/
│   ├── __init__.py
│   └── settings.py         # Configuration settings
├── input/
│   ├── images/             # Input images
│   └── video/              # Input videos
├── models/
│   ├── __init__.py
│   ├── detector_base.py    # Base detector class
│   ├── blocking_foul.py    # Blocking foul detector
│   ├── double_dribble.py   # Double dribble detector
│   ├── shot_clock.py       # Shot clock violation detector
│   ├── ten_second.py       # Ten second violation detector
│   ├── travel.py           # Traveling violation detector
│   └── weights/            # Model weights directory
├── output/                 # Detection results
├── utils/
│   ├── __init__.py
│   ├── tracking.py         # Player tracking utilities
│   ├── visualization.py    # Visualization tools
│   └── input_handler.py    # Input processing
├── main.py                 # Main application
└── test_model_loading.py   # Model testing
```

## Usage

### Basic Usage
```bash
python main.py --source input/video/game.mp4
```

### Configuration
Modify `config/settings.py` to adjust:
```python
MODEL_SETTINGS = {
    'imgsz': 1088,
    'conf_threshold': {
        'pose': 0.5,
        'ball': 0.65,
        'player': 0.7
    }
}
```

### Cloud Deployment
The project includes configuration for deployment to cloud servers (see `.idea/deployment.xml` for details).

## Requirements

### Core Dependencies
- Python 3.12
- PyTorch 2.4.1
- OpenCV 4.8.0
- Ultralytics 8.0.196
- YOLOv5 7.0.13

### Installation
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download model weights:
```bash
mkdir -p models/weights
# Download the following weights to models/weights/:
# - yolov8s-pose.pt
# - basketballModel.pt
# - best_1_27.pt
```

### Optional Dependencies
- tensorboard: For training visualization
- seaborn: For enhanced plotting

## Development

### Running Tests
```bash
python test_model_loading.py
```

### Adding New Detectors
1. Create new detector class in `models/`
2. Inherit from `BaseDetector`
3. Implement `check_violation` method
4. Register in `models/__init__.py`

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [YOLOv5](https://github.com/ultralytics/yolov5) for object detection
- [DeepSORT](https://github.com/nwojke/deep_sort) for object tracking
- [OpenCV](https://opencv.org/) for image processing

## Contact
Xiankun Jiang - 2023213655@bupt.cn

Project Link: https://github.com/jxk6575/AI_Basketball_Algorithm_System
