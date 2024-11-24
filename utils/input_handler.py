from pathlib import Path
import cv2

def get_video_files(input_dir):
    """Scan directory for video files and return numbered dict"""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_dir = Path(input_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = {}
    for idx, file in enumerate(sorted(video_dir.glob('*'))):
        if file.suffix.lower() in video_extensions:
            video_files[idx] = file
            
    return video_files

def get_image_files(input_dir):
    """Scan directory for image files and return numbered dict"""
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_dir = Path(input_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = {}
    for idx, file in enumerate(sorted(image_dir.glob('*'))):
        if file.suffix.lower() in image_extensions:
            image_files[idx] = file
            
    return image_files

def get_available_cameras():
    """Test and return available camera indices"""
    available_cameras = {}
    for i in range(5):  # Check first 5 possible camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                if i == 0:
                    available_cameras[i] = "Built-in camera"
                else:
                    available_cameras[i] = f"External camera {i}"
            cap.release()
    return available_cameras

def print_tree_structure(files_dict):
    """Print numbered tree structure of files"""
    print("\nAvailable files:")
    for idx, file in files_dict.items():
        print(f"{idx}--{file.name}")
    print()

def input_style(project_root):
    """Handle input source selection"""
    print("\nSelect input style:")
    print("0--cameras")
    print("1--images")
    print("2--videos")
    
    while True:
        try:
            choice = int(input("\nEnter choice (0-2): "))
            if choice not in (0, 1, 2):
                print("Invalid choice. Please enter 0, 1, or 2")
                continue
                
            if choice == 0:  # Camera input
                cameras = get_available_cameras()
                
                if not cameras:
                    print("No cameras detected")
                    continue
                    
                print("\nAvailable cameras:")
                for idx, name in cameras.items():
                    print(f"{idx}--{name}")
                    
                while True:
                    try:
                        camera_choice = int(input("\nSelect camera number: "))
                        if camera_choice in cameras:
                            return f"camera:{camera_choice}"
                        print("Invalid camera number")
                    except ValueError:
                        print("Please enter a valid number")
                        
            elif choice == 1:  # Image files
                image_dir = Path(project_root) / "input" / "images"
                image_files = get_image_files(image_dir)
                
                if not image_files:
                    print(f"No image files found in {image_dir}")
                    continue
                    
                print_tree_structure(image_files)
                
                while True:
                    try:
                        file_choice = int(input("Select file number: "))
                        if file_choice in image_files:
                            return str(image_files[file_choice])
                        print("Invalid file number")
                    except ValueError:
                        print("Please enter a valid number")
                        
            elif choice == 2:  # Video files
                video_dir = Path(project_root) / "input" / "video"
                video_files = get_video_files(video_dir)
                
                if not video_files:
                    print(f"No video files found in {video_dir}")
                    continue
                    
                print_tree_structure(video_files)
                
                while True:
                    try:
                        file_choice = int(input("Select file number: "))
                        if file_choice in video_files:
                            return str(video_files[file_choice])
                        print("Invalid file number")
                    except ValueError:
                        print("Please enter a valid number")
                        
        except ValueError:
            print("Please enter a valid number")