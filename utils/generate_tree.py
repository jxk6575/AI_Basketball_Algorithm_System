import os
from pathlib import Path
from datetime import datetime

def format_size(size):
    """Convert size in bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"

def generate_tree(startpath, output_file=None, exclude_dirs=None, exclude_files=None):
    """
    Generate a tree structure of the project directory and optionally save to file
    
    Args:
        startpath (str): Root directory to start from
        output_file (str): Path to output file (if None, prints to console)
        exclude_dirs (list): Directories to exclude
        exclude_files (list): File patterns to exclude
    """
    if exclude_dirs is None:
        exclude_dirs = ['.git', '__pycache__', 'venv', 'env', '.idea']
    if exclude_files is None:
        exclude_files = ['.pyc', '.pyo', '.pyd', '.so', '.dll']
    
    # Convert startpath to string if it's a Path object
    startpath = str(startpath)
    
    # Store tree lines
    tree_lines = []
    
    def should_exclude(path):
        """Check if path should be excluded"""
        path = Path(path)
        # Check directory exclusions
        if path.is_dir() and (path.name in exclude_dirs or path.name.startswith('.')):
            return True
        # Check file exclusions
        if path.is_file() and (path.suffix in exclude_files or path.name.startswith('.')):
            return True
        return False
    
    def add_to_tree(line):
        """Add a line to the tree structure"""
        tree_lines.append(line)
        if output_file is None:
            print(line)
    
    # Generate header
    header = f"Project Structure for: {Path(startpath).name}\n"
    header += "=" * (len(header) - 1) + "\n"
    header += f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    add_to_tree(header)
    
    for root, dirs, files in os.walk(startpath):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if not should_exclude(Path(root) / d)]
        
        level = root.replace(startpath, '').count(os.sep)
        indent = '│   ' * level
        
        # Add directory to tree
        if level > 0:
            add_to_tree(f"{indent[:-4]}├── {os.path.basename(root)}/")
        
        # Add files to tree
        subindent = '│   ' * (level + 1)
        
        # Sort files by extension and name
        files = sorted(files, key=lambda x: (Path(x).suffix, x))
        for i, f in enumerate(files):
            if not should_exclude(Path(root) / f):
                file_path = Path(root) / f
                size_str = ""
                
                # Add file size for large files
                if file_path.suffix in ['.pt', '.pth', '.mp4', '.avi', '.mov']:
                    size = file_path.stat().st_size
                    size_str = f" ({format_size(size)})"
                
                # Use └── for last item
                prefix = '└──' if i == len(files) - 1 else '├──'
                add_to_tree(f"{subindent[:-4]}{prefix} {f}{size_str}")
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(tree_lines))

def main():
    # Get project root directory
    project_root = Path(__file__).parent.parent
    
    # Define exclusions
    exclude_dirs = [
        '.git',
        '__pycache__',
        'venv',
        'env',
        '.idea',
        'output'    # Exclude output directory
    ]
    
    exclude_files = [
        '.pyc',
        '.pyo',
        '.pyd',
        '.so',
        '.dll',
        '.log'
    ]
    
    # Define output file path
    output_file = project_root / "project_structure.txt"
    
    # Generate tree and save to file
    generate_tree(
        project_root,
        output_file=output_file,
        exclude_dirs=exclude_dirs,
        exclude_files=exclude_files
    )

if __name__ == "__main__":
    main() 