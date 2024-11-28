import torch
import torch.nn as nn
import torchvision.transforms as T
from pathlib import Path
import numpy as np
import cv2

class LightReID(nn.Module):
    """Lightweight ReID network that doesn't require external model loading"""
    def __init__(self):
        super(LightReID, self).__init__()
        
        # Simple but effective CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Feature projection layer
        self.fc = nn.Linear(256, 128)
        
        # Set up image transforms
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ReIDModule:
    def __init__(self, model_path=None, threshold=0.5, max_features=10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.max_features = max_features
        self.feature_buffer = {}
        
        # Initialize lightweight model
        self.model = LightReID().to(self.device)
        self.model.eval()
        
    def extract_features(self, image):
        """Extract features from image patch"""
        if image is None or image.size == 0:
            return None
            
        # Preprocess image
        try:
            image = cv2.resize(image, (128, 256))
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image = image.unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(image)
                
            return features.cpu().numpy()[0]
            
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return None
            
    def update_features(self, track_id, features):
        """Update feature history for a track"""
        if features is None:
            return
            
        if track_id not in self.feature_buffer:
            self.feature_buffer[track_id] = []
            
        self.feature_buffer[track_id].append(features)
        
        # Keep only recent features
        if len(self.feature_buffer[track_id]) > self.max_features:
            self.feature_buffer[track_id].pop(0)
            
    def clean_old_tracks(self, active_tracks):
        """Remove feature history for inactive tracks"""
        current_tracks = set(active_tracks)
        stored_tracks = set(self.feature_buffer.keys())
        tracks_to_remove = stored_tracks - current_tracks
        
        for track_id in tracks_to_remove:
            self.feature_buffer.pop(track_id, None) 