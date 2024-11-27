import requests
from pathlib import Path
from config.model_config import MODEL_PATHS, MODEL_URLS
import shutil
import os
import warnings

def download_models():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for model_name, model_path in MODEL_PATHS.items():
            path = Path(model_path)
            if not path.exists() and model_name in MODEL_URLS:
                print(f"Downloading {model_name}...")
                
                # Create temp directory
                temp_dir = path.parent / 'temp'
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    if model_name == 'REID':
                        os.environ['TORCH_HOME'] = str(temp_dir)
                        import torchreid
                        model = torchreid.models.build_model(
                            name='osnet_x0_25',
                            num_classes=1000,
                            pretrained=True
                        )
                        # Move downloaded weights
                        weight_file = next(temp_dir.rglob('*.pth'))
                        shutil.move(str(weight_file), str(path))
                    
                    print(f"Downloaded {model_name}")
                    
                except Exception as e:
                    print(f"Error downloading {model_name}: {e}")
                finally:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    download_models()