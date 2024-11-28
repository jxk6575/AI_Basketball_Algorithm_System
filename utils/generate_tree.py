import os
from pathlib import Path
from datetime import datetime

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
        """Add line to tree structure"""
        tree_lines.append(line)
        if not output_file:
            print(line)
    
    def print_tree(startpath, prefix=''):
        """Recursively print directory tree"""
        if should_exclude(startpath):
            return
            
        # Print current item
        path = Path(startpath)
        add_to_tree(f"{prefix}├── {path.name}")
        
        # Handle directory contents
        if path.is_dir():
            # Get and sort directory contents
            contents = sorted(list(path.iterdir()), 
                           key=lambda x: (x.is_file(), x.name.lower()))
            
            # Process each item
            for i, item in enumerate(contents):
                if should_exclude(item):
                    continue
                    
                # Determine prefix for next level
                is_last = i == len(contents) - 1
                new_prefix = prefix + ('    ' if is_last else '│   ')
                
                # Recursively print item
                print_tree(item, new_prefix)
    
    # Start generating tree from root
    root = Path(startpath)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add header
    add_to_tree(f"Project Structure for: {root.name}")
    add_to_tree("=" * (20 + len(root.name)))
    add_to_tree(f"Generated at: {timestamp}\n")
    
    # Generate tree
    print_tree(startpath)
    add_to_tree("\n")  # Add final newline
    
    # Save to file if output_file is specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(tree_lines))
            print(f"\nTree structure saved to: {output_path}")
        except Exception as e:
            print(f"\nError saving tree structure: {e}")

def main():
    # Get project root (assuming this script is in utils/)
    project_root = Path(__file__).parent.parent
    
    # Define exclusions
    exclude_dirs = [
        '.git',
        '__pycache__',
        'venv',
        'env',
        '.idea',
        'weights',  # Exclude model weights directory
        'output'    # Exclude output directory
    ]
    
    exclude_files = [
        '.pyc',
        '.pyo',
        '.pyd',
        '.so',
        '.dll',
        '.log',
        '.mp4',
        '.avi',
        '.mov'
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