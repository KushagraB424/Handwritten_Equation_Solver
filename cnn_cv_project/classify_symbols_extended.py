import torch
import torch.nn as nn
import cv2
import numpy as np
import pickle
from pathlib import Path

class ImprovedSymbolCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedSymbolCNN, self).__init__()
        
        # Convolutional layers with residual connections
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.4)
        )
        
        # Residual blocks
        self.res_block1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128)
        )
        
        self.res_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128)
        )
        
        # Global Average Pooling and fully connected layers
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = nn.MaxPool2d(2)(x)
        
        x = self.conv2(x)
        x = nn.MaxPool2d(2)(x)
        
        x = self.conv3(x)
        x = nn.MaxPool2d(2)(x)
        
        # Residual blocks
        identity = x
        x = self.res_block1(x)
        x += identity
        x = nn.ReLU()(x)
        
        identity = x
        x = self.res_block2(x)
        x += identity
        x = nn.ReLU()(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class SymbolClassifier:
    def __init__(self, model_path='equation_solver_model_extended.pt', 
                 label_encoder_path='label_encoder_extended.pkl',
                 label_mapping_path='label_mapping_extended.pkl'):
        """Initialize the symbol classifier with extended model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load label mapping
        with open(label_mapping_path, 'rb') as f:
            self.label_mapping = pickle.load(f)
        
        # Initialize model
        self.model = ImprovedSymbolCNN(num_classes=len(self.label_encoder.classes_))
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.image_size = checkpoint.get('image_size', (64, 64))
        
        print(f"Model loaded successfully!")
        print(f"Device: {self.device}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print(f"Classes: {self.label_encoder.classes_}")
    
    def preprocess_image(self, image):
        """Preprocess image for classification"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        image = cv2.resize(image, self.image_size)
        
        # Apply threshold
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Normalize and convert to tensor
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        
        return image.to(self.device)
    
    def classify(self, image, return_probabilities=False):
        """Classify a single symbol image"""
        # Preprocess
        processed_image = self.preprocess_image(image)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(processed_image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Decode label
        predicted_label = self.label_encoder.inverse_transform([predicted.item()])[0]
        
        # Map to final output (pi -> np.pi, e -> np.e)
        final_label = self.label_mapping.get(predicted_label, predicted_label)
        
        if return_probabilities:
            probs_dict = {}
            for idx, prob in enumerate(probabilities[0].cpu().numpy()):
                label = self.label_encoder.inverse_transform([idx])[0]
                mapped_label = self.label_mapping.get(label, label)
                probs_dict[mapped_label] = float(prob)
            return final_label, confidence.item(), probs_dict
        
        return final_label, confidence.item()
    
    def classify_batch(self, images):
        """Classify multiple symbol images"""
        results = []
        for image in images:
            label, confidence = self.classify(image)
            results.append((label, confidence))
        return results

def main():
    """Demo classification"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python classify_symbols_extended.py <image_path>")
        print("Example: python classify_symbols_extended.py test_symbol.png")
        return
    
    image_path = sys.argv[1]
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Initialize classifier
    classifier = SymbolClassifier()
    
    # Classify
    label, confidence, probabilities = classifier.classify(image, return_probabilities=True)
    
    print(f"\nClassification Results:")
    print(f"Predicted: {label}")
    print(f"Confidence: {confidence:.4f}")
    
    print(f"\nTop 5 predictions:")
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
    for symbol, prob in sorted_probs:
        print(f"  {symbol}: {prob:.4f}")

if __name__ == "__main__":
    main()
