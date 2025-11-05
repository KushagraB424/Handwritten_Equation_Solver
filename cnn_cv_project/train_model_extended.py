import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
import pickle
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

class SymbolDataset(Dataset):
    def __init__(self, images, labels, transform=None, augment=False):
        self.images = images
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
        self.augment = augment
        
        # Augmentation transforms
        self.aug_transforms = transforms.Compose([
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image for torchvision transforms
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        
        if self.transform:
            image = self.transform(image)
        
        if self.augment:
            image = self.aug_transforms(image)
        
        return image, label

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

def load_dataset(data_root: str = "data_root", image_size=(64, 64)):
    """Load and preprocess the dataset from data_root, excluding 'equals'"""
    data_path = Path(data_root)
    images = []
    labels = []
    
    # Get all subdirectories in data_root
    all_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    print("Loading dataset from data_root...")
    print(f"Found directories: {[d.name for d in all_dirs]}")
    
    for symbol_dir in tqdm(all_dirs, desc="Loading symbols"):
        symbol_class = symbol_dir.name
        
        # Skip 'equals' and 'dataset' directories
        if symbol_class in ['equals', 'dataset']:
            print(f"Skipping '{symbol_class}' directory")
            continue
        
        count = 0
        # Set limit for pi and e to 500 images each
        max_images = 500 if symbol_class in ['pi', 'e'] else None
        
        # Load both .png and .jpg files
        image_files = list(symbol_dir.glob("*.png")) + list(symbol_dir.glob("*.jpg"))
        
        for img_path in image_files:
            # Stop if we've reached the limit for pi or e
            if max_images is not None and count >= max_images:
                break
                
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Basic preprocessing
                img = cv2.resize(img, image_size)
                _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                images.append(img)
                labels.append(symbol_class)
                count += 1
        
        if count > 0:
            print(f"Loaded {count} images for '{symbol_class}'")
    
    return images, labels

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    images, labels = load_dataset()
    
    if len(images) == 0:
        raise ValueError("No training images found!")
    
    print(f"\nTotal loaded: {len(images)} images")
    
    # Count labels
    from collections import Counter
    label_counts = Counter(labels)
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    encoded_labels = label_encoder.transform(labels)
    
    print(f"\nLabel classes: {label_encoder.classes_}")
    
    # Create mapping for pi -> np.pi and e -> np.e
    label_mapping = {}
    for label in label_encoder.classes_:
        if label == 'pi':
            label_mapping[label] = 'np.pi'
        elif label == 'e':
            label_mapping[label] = 'np.e'
        else:
            label_mapping[label] = label
    
    # Save label encoder and mapping
    with open('label_encoder_extended.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    with open('label_mapping_extended.pkl', 'wb') as f:
        pickle.dump(label_mapping, f)
    
    print(f"\nLabel mapping:")
    for k, v in sorted(label_mapping.items()):
        print(f"  {k} -> {v}")
    
    # Create datasets
    dataset = SymbolDataset(images=images, labels=encoded_labels)
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Enable augmentation for training set
    train_dataset.dataset.augment = True
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    print(f"\nDataset splits:")
    print(f"  Training: {len(train_dataset)} images")
    print(f"  Validation: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    
    # Initialize model
    model = ImprovedSymbolCNN(num_classes=len(label_encoder.classes_))
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training parameters
    num_epochs = 100
    early_stopping_patience = 10
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save best model
            torch.save({
                'state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'image_size': (64, 64)
            }, 'equation_solver_model_extended.pt')
            print(f"✓ Model saved with val_loss: {val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print("Early stopping triggered")
                break
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100 * test_correct / test_total
    
    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.2f}%")
    
    # Save test results
    with open('test_results_extended.txt', 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"\nLabel mapping:\n")
        for k, v in sorted(label_mapping.items()):
            f.write(f"  {k} -> {v}\n")
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)
    print(f"Model saved as: equation_solver_model_extended.pt")
    print(f"Label encoder saved as: label_encoder_extended.pkl")
    print(f"Label mapping saved as: label_mapping_extended.pkl")

if __name__ == "__main__":
    main()
