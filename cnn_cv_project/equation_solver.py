import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import sympy as sp
from typing import List, Tuple, Dict, Union
import pickle

class SymbolDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class SymbolCNN(nn.Module):
    def __init__(self, num_classes):
        super(SymbolCNN, self).__init__()
        
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

class EquationSolver:
    def __init__(self, data_root: str = "data_root", use_extended_model: bool = True):
        """
        Initialize the equation solver with paths and models
        Args:
            data_root: Path to training data directory containing symbol folders
            use_extended_model: If True, use extended model with π and e support
        """
        self.data_root = Path(data_root)
        self.label_encoder = LabelEncoder()
        self.image_size = (64, 64)  # Increased size for CNN
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.model = None  # Will be initialized during training
        self.use_extended_model = use_extended_model
        self.label_mapping = {}  # For pi -> np.pi, e -> np.e mapping
        
        if use_extended_model:
            self.symbol_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                 'x', 'y', 'z', 'plus', 'minus', 'mul', 'div', 'dec', 'pi', 'e']
        else:
            self.symbol_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                 'x', 'y', 'z', 'plus', 'minus', 'mul', 'div', 'equals']
        
        self.symbol_map = {
            'plus': '+',
            'minus': '-',
            'mul': '*',
            'div': '/',
            'equals': '=',
            'equal': '=',
            'dec': '.'
        }
        self.last_left_expression = ""
        self.last_right_expression = ""
        self.last_equation = ""
        
    def load_training_data(self) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load training images from data_root folders
        Returns:
            images: List of preprocessed grayscale images
            labels: List of corresponding labels
        """
        images = []
        labels = []
        
        for symbol_class in self.symbol_classes:
            symbol_dir = self.data_root / str(symbol_class)
            if not symbol_dir.exists():
                continue
                
            for img_path in symbol_dir.glob("*.png"):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Basic preprocessing
                    img = cv2.resize(img, self.image_size)
                    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                    images.append(img)
                    labels.append(symbol_class)
        
        return images, labels

    def extract_features(self, img: np.ndarray) -> np.ndarray:
        """
        Extract combined SIFT and HOG features
        Args:
            img: Input grayscale image
        Returns:
            feature_vector: Combined feature vector
        """
        if img is None:
            return np.zeros(424)  # Combined feature length
            
        # Normalize image
        img = cv2.resize(img, (64, 64))
        img = cv2.equalizeHist(img)
        
        # 1. SIFT Features
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        
        if descriptors is None or len(descriptors) == 0:
            sift_features = np.zeros(256)
        else:
            # Mean of SIFT descriptors
            mean_desc = np.mean(descriptors, axis=0)
            
            # Spatial histogram of keypoint locations
            h, w = img.shape
            spatial_hist = np.zeros(128)
            for kp in keypoints:
                x, y = kp.pt
                bin_x = int(4 * x / w)
                bin_y = int(4 * y / h)
                if 0 <= bin_x < 4 and 0 <= bin_y < 4:
                    bin_idx = bin_y * 4 + bin_x
                    spatial_hist[bin_idx] = 1
                    
            sift_features = np.concatenate([mean_desc, spatial_hist])
            
        # 2. HOG Features
        # Calculate HOG features
        win_size = (64, 64)
        cell_size = (8, 8)
        block_size = (16, 16)
        block_stride = (8, 8)
        num_bins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
        hog_features = hog.compute(img).flatten()
        
        # Combine all features
        feature_vector = np.concatenate([sift_features, hog_features])
        
        # Normalize feature vector
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / norm
            
        return feature_vector

    def _clean_expression(self, expr: str) -> str:
        """Normalize an expression string by removing duplicate operators and leading noise."""
        if not expr:
            return ""

        expr = expr.strip()
        for bad in ['++', '--', '**', '//', '+-', '-+', '*+', '+*', '/+', '+/']:
            expr = expr.replace(bad, bad[0])

        return expr.lstrip('+-*/')

    def _apply_implicit_multiplication(self, expr: str) -> str:
        """Insert explicit multiplication where needed (e.g., 2x -> 2*x, 2np.pi -> 2*np.pi)."""
        if not expr:
            return expr

        result_chars = []
        i = 0
        while i < len(expr):
            char = expr[i]
            
            # Check for np.pi or np.e
            if i + 5 <= len(expr) and expr[i:i+5] in ['np.pi', 'np.e']:
                # Add multiplication before np.pi or np.e if preceded by digit or )
                if i > 0 and (expr[i-1].isdigit() or expr[i-1] == ')'):
                    result_chars.append('*')
                
                if expr[i:i+5] == 'np.pi':
                    result_chars.append('np.pi')
                    i += 5
                else:  # np.e
                    result_chars.append('np.e')
                    i += 4
                continue
            
            # Add multiplication between digit and letter
            if i > 0:
                prev_char = expr[i - 1]
                if char.isalpha() and prev_char.isdigit():
                    result_chars.append('*')
            
            result_chars.append(char)
            i += 1
        
        return ''.join(result_chars)

    def parse_expression_from_image(self, img: np.ndarray) -> str:
        """Parse a single-side expression image into a cleaned string without an equals sign."""
        if img is None:
            return ""

        symbols = self.segment_equation(img)
        parts: List[str] = []

        for symbol in symbols:
            predicted_symbol = self.predict_symbol(symbol)
            if predicted_symbol is None:
                continue

            mapped = self.symbol_map.get(predicted_symbol, predicted_symbol)
            if mapped == '=':
                # Ignore equals when parsing individual sides
                continue
            parts.append(mapped)

        expression = ''.join(parts)
        return self._clean_expression(expression)

    def solve_symbolic_from_strings(self, left_expr_str: str, right_expr_str: str) -> Dict[str, Union[float, str]]:
        """Solve an equation given left and right expression strings."""
        left_expr_str = left_expr_str or ""
        right_expr_str = right_expr_str or ""

        if not left_expr_str:
            raise ValueError("Left side could not be parsed")
        if not right_expr_str:
            raise ValueError("Right side could not be parsed")

        left_for_sympy = self._apply_implicit_multiplication(left_expr_str)
        right_for_sympy = self._apply_implicit_multiplication(right_expr_str)
        
        # Replace np.pi and np.e with sympy equivalents
        left_for_sympy = left_for_sympy.replace('np.pi', 'pi').replace('np.e', 'E')
        right_for_sympy = right_for_sympy.replace('np.pi', 'pi').replace('np.e', 'E')

        x, y, z = sp.symbols('x y z')

        try:
            left_expr = sp.sympify(left_for_sympy)
            right_expr = sp.sympify(right_for_sympy)
        except Exception as exc:
            raise ValueError(f"Invalid equation format: {exc}") from exc

        equation = sp.Eq(left_expr, right_expr)
        vars_in_eq = [var for var in (x, y, z) if equation.has(var)]

        if not vars_in_eq:
            raise ValueError("No variable found to solve for")

        var_to_solve = vars_in_eq[0]

        try:
            solutions = sp.solve(equation, var_to_solve)
        except Exception as exc:
            raise ValueError(f"Solver error: {exc}") from exc

        if not solutions:
            raise ValueError("No real solution found")

        first_solution = solutions[0]

        try:
            solution_value: Union[float, str] = float(first_solution)
        except Exception:
            solution_value = str(first_solution)

        return {str(var_to_solve): solution_value}

    def solve_from_sides(self, left_img: np.ndarray, right_img: np.ndarray) -> Dict[str, Union[str, Dict[str, Union[float, str]]]]:
        """Parse and solve an equation represented as separate left/right images."""
        left_expr = self.parse_expression_from_image(left_img)
        right_expr = self.parse_expression_from_image(right_img)

        solution = self.solve_symbolic_from_strings(left_expr, right_expr)

        equation_str = f"{left_expr}={right_expr}"
        self.last_left_expression = left_expr
        self.last_right_expression = right_expr
        self.last_equation = equation_str

        return {
            'left': left_expr,
            'right': right_expr,
            'equation': equation_str,
            'solution': solution
        }

    def train_classifier(self, batch_size=32, num_epochs=50) -> None:
        """Train the CNN classifier on the symbol dataset"""
        print("Loading training data...")
        images, labels = self.load_training_data()
        
        if len(images) == 0:
            raise ValueError("No training images found!")
            
        print(f"Processing {len(images)} images...")
        
        # Encode labels
        self.label_encoder.fit(self.symbol_classes)
        encoded_labels = self.label_encoder.transform(labels)
        
        # Create dataset and dataloader
        dataset = SymbolDataset(
            images=images,
            labels=encoded_labels,
            transform=self.transform
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        self.model = SymbolCNN(num_classes=len(self.symbol_classes))
        self.model = self.model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        print("Training CNN...")
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            
            epoch_loss = running_loss / len(dataloader)
            accuracy = 100 * correct / total
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Loss: {epoch_loss:.4f}, "
                      f"Accuracy: {accuracy:.2f}%")
        
    def save_model(self, model_path: str = "equation_solver_model.pt") -> None:
        """Save the trained model and label encoder"""
        if self.model is None:
            raise ValueError("No trained model to save!")
            
        # Save CNN model
        model_state = {
            'state_dict': self.model.state_dict(),
            'image_size': self.image_size
        }
        torch.save(model_state, model_path)
        
        # Save label encoder separately
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
            
    def load_model(self, model_path: str = None, label_encoder_path: str = None, label_mapping_path: str = None) -> None:
        """Load a trained model and label encoder"""
        # Set default paths based on model type
        if model_path is None:
            model_path = "equation_solver_model_extended.pt" if self.use_extended_model else "equation_solver_model.pt"
        if label_encoder_path is None:
            label_encoder_path = "label_encoder_extended.pkl" if self.use_extended_model else "label_encoder.pkl"
        if label_mapping_path is None and self.use_extended_model:
            label_mapping_path = "label_mapping_extended.pkl"
        
        # Load label encoder FIRST to get the actual classes
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Update symbol_classes from the loaded label encoder
        self.symbol_classes = list(self.label_encoder.classes_)
        
        # Load model state
        model_state = torch.load(model_path, map_location=self.device)
        self.image_size = model_state['image_size']
        
        # Initialize and load model with correct number of classes
        self.model = SymbolCNN(num_classes=len(self.label_encoder.classes_))
        self.model.load_state_dict(model_state['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load label mapping if using extended model
        if self.use_extended_model and label_mapping_path:
            try:
                with open(label_mapping_path, 'rb') as f:
                    self.label_mapping = pickle.load(f)
            except FileNotFoundError:
                print("Warning: Label mapping file not found, using default mapping")
                self.label_mapping = {'pi': 'np.pi', 'e': 'np.e'}

    def segment_equation(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Segment an equation image into individual symbols using improved preprocessing and connected components
        Args:
            img: Input equation image
        Returns:
            List of segmented symbol images
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        # Store original size for debug visualization
        debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # 1. Enhanced preprocessing
        # Apply bilateral filter to reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Adaptive thresholding to handle varying illumination
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # 2. Morphological operations
        # Vertical kernel to help connect equals sign
        kernel_vertical = np.ones((5,1), np.uint8)
        # Horizontal kernel for general cleanup
        kernel = np.ones((3,3), np.uint8)
        
        # First connect vertical components (like equals sign)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_vertical)
        
        # Then general cleanup
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 3. Connected Components Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Filter components based on size and position
        min_size = 30  # Reduced minimum component size to catch smaller symbols
        max_size = img.shape[0] * img.shape[1] * 0.3  # Maximum 30% of image
        valid_components = []
        
        for i in range(1, num_labels):  # Skip background (label 0)
            x, y, w, h, area = stats[i]
            if min_size < area < max_size:
                aspect_ratio = w / h if h > 0 else float('inf')
                # More lenient aspect ratio check for special symbols like equals
                if 0.05 < aspect_ratio < 15:  # Wider range for aspect ratio
                    valid_components.append((x, y, w, h, area))
        
        # Sort components left-to-right
        valid_components.sort(key=lambda x: x[0])
        
        # 4. Extract and process symbols
        symbols = []
        prev_x_end = 0
        merge_distance = 20  # Increased distance threshold for merging components (especially for equal signs)
        
        current_group = []
        for x, y, w, h, area in valid_components:
            if x - prev_x_end <= merge_distance:
                # Add to current group
                current_group.append((x, y, w, h))
            else:
                # Process previous group if exists
                if current_group:
                    # Find bounding box of group
                    min_x = min(comp[0] for comp in current_group)
                    min_y = min(comp[1] for comp in current_group)
                    max_x = max(comp[0] + comp[2] for comp in current_group)
                    max_y = max(comp[1] + comp[3] for comp in current_group)
                    group_w = max_x - min_x
                    group_h = max_y - min_y
                    
                    # Extract symbol
                    pad = 4  # Add padding
                    y_start = max(0, min_y - pad)
                    y_end = min(gray.shape[0], max_y + pad)
                    x_start = max(0, min_x - pad)
                    x_end = min(gray.shape[1], max_x + pad)
                    
                    # Draw bounding box in debug image
                    cv2.rectangle(debug_img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                    
                    symbol = gray[y_start:y_end, x_start:x_end]
                    
                    # Ensure black text on white background
                    if np.mean(symbol) < 127:
                        symbol = 255 - symbol
                    
                    # Resize maintaining aspect ratio
                    target_size = 64
                    aspect = group_w / group_h
                    if aspect > 1:
                        new_w = target_size
                        new_h = int(target_size / aspect)
                    else:
                        new_h = target_size
                        new_w = int(target_size * aspect)
                    
                    symbol = cv2.resize(symbol, (new_w, new_h))
                    
                    # Pad to square
                    final_symbol = np.ones((target_size, target_size), dtype=np.uint8) * 255
                    y_offset = (target_size - new_h) // 2
                    x_offset = (target_size - new_w) // 2
                    final_symbol[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = symbol
                    
                    symbols.append(final_symbol)
                
                # Start new group
                current_group = [(x, y, w, h)]
            
            prev_x_end = x + w
        
        # Process last group
        if current_group:
            min_x = min(comp[0] for comp in current_group)
            min_y = min(comp[1] for comp in current_group)
            max_x = max(comp[0] + comp[2] for comp in current_group)
            max_y = max(comp[1] + comp[3] for comp in current_group)
            
            pad = 4
            y_start = max(0, min_y - pad)
            y_end = min(gray.shape[0], max_y + pad)
            x_start = max(0, min_x - pad)
            x_end = min(gray.shape[1], max_x + pad)
            
            # Draw bounding box in debug image
            cv2.rectangle(debug_img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            
            symbol = gray[y_start:y_end, x_start:x_end]
            if np.mean(symbol) < 127:
                symbol = 255 - symbol
                
            # Resize and pad as before
            aspect = (max_x - min_x) / (max_y - min_y)
            target_size = 64
            if aspect > 1:
                new_w = target_size
                new_h = int(target_size / aspect)
            else:
                new_h = target_size
                new_w = int(target_size * aspect)
                
            symbol = cv2.resize(symbol, (new_w, new_h))
            final_symbol = np.ones((target_size, target_size), dtype=np.uint8) * 255
            y_offset = (target_size - new_h) // 2
            x_offset = (target_size - new_w) // 2
            final_symbol[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = symbol
            symbols.append(final_symbol)
        
        # Save debug visualization
        cv2.imwrite('debug_segmentation.png', debug_img)
        
        # Also save individual symbols for debugging
        for i, symbol in enumerate(symbols):
            cv2.imwrite(f'debug_symbol_{i}.png', symbol)
            
        return symbols

    def predict_symbol(self, symbol_img: np.ndarray) -> str:
        """
        Predict the class of a single symbol using the trained CNN
        Args:
            symbol_img: Preprocessed symbol image
        Returns:
            Predicted symbol class (with label mapping applied)
        """
        if self.model is None:
            raise ValueError("Model not trained! Please train the model first.")
            
        # Ensure grayscale
        if len(symbol_img.shape) == 3:
            img = cv2.cvtColor(symbol_img, cv2.COLOR_BGR2GRAY)
        else:
            img = symbol_img.copy()
            
        # Preprocess the image
        img = cv2.resize(img, self.image_size)
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Convert to tensor and normalize
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            confidence = confidence.item()
            prediction = prediction.item()
        
        # Only predict if confidence is high enough
        if confidence > 0.7:  # Higher threshold for CNN predictions
            predicted_symbol = self.label_encoder.inverse_transform([prediction])[0]
            
            # Apply label mapping if using extended model (pi -> np.pi, e -> np.e)
            if self.use_extended_model and predicted_symbol in self.label_mapping:
                mapped_symbol = self.label_mapping[predicted_symbol]
                print(f"Predicted symbol: {predicted_symbol} -> {mapped_symbol} (confidence: {confidence:.2f})")
                return mapped_symbol
            
            # Convert 'z' predictions to 'equals' for backward compatibility
            if predicted_symbol == 'z':
                predicted_symbol = 'equals'
            
            print(f"Predicted symbol: {predicted_symbol} (confidence: {confidence:.2f})")
            return predicted_symbol
        
        print(f"Low confidence prediction ({confidence:.2f}), skipping symbol")
        return None

    def parse_equation(self, equation_img: np.ndarray) -> str:
        """
        Convert equation image to string representation
        Args:
            equation_img: Input image containing the equation
        Returns:
            String representation of the equation
        """
        symbols = self.segment_equation(equation_img)
        equation_parts = []
        
        for symbol in symbols:
            predicted_symbol = self.predict_symbol(symbol)
            if predicted_symbol is None:
                continue
                
            # Convert symbol names to actual operators
            equation_parts.append(self.symbol_map.get(predicted_symbol, predicted_symbol))
        
        if not equation_parts:
            raise ValueError("Could not parse any symbols from the equation!")
            
        equation_str = ''.join(equation_parts)

        if '=' in equation_str:
            left_side, right_side = equation_str.split('=', 1)
            left_side = self._clean_expression(left_side)
            right_side = self._clean_expression(right_side)
            equation_str = f"{left_side}={right_side}"
        else:
            equation_str = self._clean_expression(equation_str)
        
        print(f"Parsed equation string: {equation_str}")
        # Store the last equation for the web interface
        self.last_equation = equation_str
        return equation_str

    def solve_equation(self, equation_img: np.ndarray) -> Dict[str, float]:
        """
        Solve the equation in the image
        Args:
            equation_img: Input image containing the equation
        Returns:
            Dictionary mapping variables to their solutions
        """
        equation_str = self.parse_equation(equation_img)
        print(f"Parsed equation: {equation_str}")
        
        # Handle case where there's no equals sign
        if '=' not in equation_str:
            # Treat as expression and simplify
            x, y, z = sp.symbols('x y z')
            expr = sp.sympify(equation_str)
            print(f"Expression (no equals sign): {expr}")
            return {'expression': str(sp.simplify(expr))}
        
        # Split into left and right sides if equals sign present
        left, right = equation_str.split('=')
        
        # Convert to SymPy expression
        x, y, z = sp.symbols('x y z')
        # Add multiplication symbol between numbers and variables
        left = ''.join([f"*{c}" if i > 0 and c.isalpha() and left[i-1].isdigit() else c for i, c in enumerate(left)])
        right = ''.join([f"*{c}" if i > 0 and c.isalpha() and right[i-1].isdigit() else c for i, c in enumerate(right)])
        
        left_expr = sp.sympify(left)
        right_expr = sp.sympify(right)
        
        # Solve equation
        equation = sp.Eq(left_expr, right_expr)
        
        # Detect which variable to solve for
        vars_in_eq = [var for var in (x, y, z) if equation.has(var)]
        if not vars_in_eq:
            raise ValueError("No variables found in equation!")
            
        var_to_solve = vars_in_eq[0]  # Take first variable found
        solution = sp.solve(equation, var_to_solve)
        
        if not solution:
            raise ValueError("No solution found!")
            
        # Convert to float if possible
        try:
            solution_value = float(solution[0])
        except TypeError:
            solution_value = str(solution[0])
            
        return {str(var_to_solve): solution_value}

def main():
    # Create and train the model
    solver = EquationSolver()
    
    try:
        # Try to load a pre-trained model
        solver.load_model()
        print("Loaded pre-trained model.")
    except FileNotFoundError:
        print("Training new model...")
        solver.train_classifier()
        solver.save_model()
        print("Model trained and saved.")
    
    # Example usage (uncomment and modify path as needed):
    # equation_img = cv2.imread('path_to_equation_image.png', cv2.IMREAD_GRAYSCALE)
    # solution = solver.solve_equation(equation_img)
    # print(f"Solution: {solution}")

if __name__ == "__main__":
    main()