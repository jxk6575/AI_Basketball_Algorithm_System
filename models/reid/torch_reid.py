import torchreid
import torch
import numpy as np
import cv2
from pathlib import Path
import os
import shutil
import gdown


class TorchReID:
    def __init__(self):
        # Get project root and weights directory
        project_root = Path(__file__).parent.parent.parent
        weights_dir = project_root / 'models' / 'weights'
        weights_dir.mkdir(parents=True, exist_ok=True)

        # Define weight paths and URL
        weights_path = weights_dir / 'osnet_x0_25.pth'
        model_url = 'https://drive.google.com/uc?id=1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs'

        try:
            if weights_path.exists():
                print(f"Loading ReID model from {weights_path}")
            else:
                print("Downloading ReID model weights...")
                # Create temporary directory
                temp_dir = weights_dir / 'temp'
                temp_dir.mkdir(exist_ok=True)

                # Download weights using gdown
                temp_path = temp_dir / 'osnet_x0_25.pth'
                gdown.download(model_url, str(temp_path), quiet=False)

                # Move to final location
                shutil.move(str(temp_path), str(weights_path))
                shutil.rmtree(temp_dir)
                print(f"ReID model weights saved to {weights_path}")

            # Initialize model
            self.model = torchreid.models.build_model(
                name='osnet_x0_25',
                num_classes=1000,
                pretrained=False
            )

            # Load weights
            self.model.load_state_dict(torch.load(str(weights_path)))
            self.model.eval()
            print(f"ReID model loaded successfully")

            # Initialize other parameters
            self.height = 256
            self.width = 128
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            self.feature_cache = {}
            self.max_cache_size = 1000

        except Exception as e:
            print(f"Error initializing ReID model: {e}")
            raise

    def preprocess(self, image):
        """Preprocess image for model input"""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize
            image = cv2.resize(image, (self.width, self.height))

            # Convert to float and normalize
            image = image.astype(np.float32) / 255.0
            image = (image - self.mean) / self.std

            # Convert to torch tensor and add batch dimension
            image = torch.from_numpy(image).float()
            image = image.permute(2, 0, 1).unsqueeze(0)
            return image
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

    def extract_features(self, image):
        """Extract features from image"""
        with torch.no_grad():
            try:
                # Generate image hash for caching
                image_hash = hash(image.tobytes())

                # Return cached features if available
                if image_hash in self.feature_cache:
                    return self.feature_cache[image_hash]

                # Preprocess image
                input_tensor = self.preprocess(image)
                if input_tensor is None:
                    return np.zeros(512)  # Return zero features if preprocessing fails

                # Extract features
                features = self.model(input_tensor)
                features = features.cpu().numpy()[0]

                # Normalize features
                norm = np.linalg.norm(features)
                if norm > 0:
                    features = features / norm

                # Cache features
                self.feature_cache[image_hash] = features

                # Manage cache size
                if len(self.feature_cache) > self.max_cache_size:
                    # Remove oldest entries
                    keys = list(self.feature_cache.keys())
                    for old_key in keys[:100]:
                        del self.feature_cache[old_key]

                return features

            except Exception as e:
                print(f"Feature extraction error: {e}")
                return np.zeros(512)  # Return zero features on error

    def compute_similarity(self, feat1, feat2):
        """Compute cosine similarity between features"""
        if feat1 is None or feat2 is None:
            return 0.0
        return np.dot(feat1, feat2)

    def clear_cache(self):
        """Clear the feature cache"""
        self.feature_cache.clear()

    def get_model_info(self):
        """Get model information for debugging"""
        return {
            "model_name": "osnet_x0_25",
            "input_size": f"{self.height}x{self.width}",
            "feature_dim": 512,
            "cache_size": len(self.feature_cache)
        }