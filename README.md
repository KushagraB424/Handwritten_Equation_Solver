# Handwritten Equation Solver

A Python application that uses classical computer vision and machine learning to recognize and solve handwritten mathematical equations.

## Features

- Preprocesses images of handwritten equations using advanced computer vision techniques
- Implements HOG (Histogram of Oriented Gradients) and pixel intensity features
- Uses K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) with RBF kernel for classification
- Segments and recognizes individual digits and operators with high accuracy
- Solves linear equations and arithmetic expressions
- Visualizes the recognition process

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd handwritten-equation-solver
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your handwritten equation image in the project directory.
2. Update the `image_path` in `equation_solver.py` to point to your image.
3. Run the solver:
   ```bash
   python equation_solver.py
   ```

## Example

For an image named `equation.png` containing a handwritten equation like "2x + 5 = 15", the output will be:

```
Solution: x = 5
```

## How It Works

### Image Preprocessing
1. **Input Image Handling**: Accepts grayscale or color images of handwritten equations
2. **Binarization**: Converts the image to binary using Otsu's thresholding
3. **Character Segmentation**:
   - Applies contour detection to isolate individual characters
   - Performs size normalization while maintaining aspect ratio
   - Centers characters in a 32x32 pixel canvas for consistent processing

### Feature Extraction
1. **HOG Features**:
   - Computes gradient magnitude and orientation using Sobel operators
   - Divides the image into 8x8 pixel cells
   - Calculates 9-bin histograms of gradient orientations per cell
   - Normalizes histograms in 2x2 cell blocks using L2-Hys normalization
   - Results in a 324-dimensional feature vector (4 blocks × 4 cells × 9 bins)

2. **Pixel Intensity Features**:
   - Flattens the 32x32 normalized image into a 1024-dimensional vector
   - Normalizes pixel values to [0, 1] range

### Model Architecture
The system employs two complementary models:
1. **K-Nearest Neighbors (KNN)**:
   - Non-parametric method for classification
   - Uses k=3 neighbors with uniform weights
   - Effective for capturing local patterns in handwritten digits

2. **Support Vector Machine with RBF Kernel (SVM-RBF)**:
   - Implements a non-linear decision boundary
   - Uses Radial Basis Function kernel for better separability
   - Optimized hyperparameters for character recognition

### Equation Processing
1. **Character Recognition**:
   - Combines HOG and pixel features into a single feature vector
   - Uses the trained models to predict individual characters
   - Applies post-processing to improve recognition accuracy

2. **Equation Solving**:
   - Parses the sequence of recognized characters
   - Validates mathematical expressions
   - Solves linear equations and arithmetic expressions

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- scikit-image
- TensorFlow
- SymPy
- Matplotlib

## Performance Considerations

- The system achieves optimal results with clear, well-separated characters
- Works best with equations written on a clean, high-contrast background
- Performance may vary with different writing styles and equation complexity
- For improved accuracy, consider training on a larger dataset of handwritten equations

## Implementation Details

### Feature Extraction Pipeline
1. **Input Normalization**:
   ```python
   def resize_keep_aspect(img, target=32):
       # Resizes image while maintaining aspect ratio
       # Pads with zeros to reach target dimensions
   ```

2. **HOG Descriptor**:
   ```python
   def hog_descriptor(img32: np.ndarray) -> np.ndarray:
       # Implements HOG feature extraction
       # Returns a 324-dimensional feature vector
   ```

3. **Intensity Features**:
   ```python
   def intensity_features(img32: np.ndarray) -> np.ndarray:
       # Extracts normalized pixel intensities
       # Returns a 1024-dimensional feature vector
   ```

### Model Training
- The models are trained on a dataset of handwritten digits and operators
- Feature vectors are normalized before training
- Cross-validation is used to optimize model parameters

## Future Improvements

- Expand character set to include more mathematical symbols
- Implement equation structure analysis for better parsing
- Add support for multi-line equations
- Incorporate deep learning models for improved recognition accuracy


# Extende with π ane e support

A deep learning project for computer vision tasks using Convolutional Neural Networks (CNN).

## Project Overview
This project implements various CNN architectures for image classification, object detection, or other computer vision tasks. It's designed to be modular and easy to extend for different use cases.

## Features
- Support for multiple CNN architectures
- Data preprocessing and augmentation
- Model training and evaluation
- Pretrained model support (e.g., ResNet, VGG, etc.)
- Visualization tools for model performance

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cnn-cv-project.git
   cd cnn-cv-project
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Linux/Mac
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Prepare your dataset in the appropriate directory structure
2. Configure the model parameters in `config.py` (if available)
3. Run the training script:
   ```bash
   python train.py
   ```
4. Evaluate the model:
   ```bash
   python evaluate.py
   ```

## CNN Architecture

The model is built using PyTorch and features a custom CNN architecture with residual connections for improved training stability and performance.

### Model Architecture

1. **Input Layer**
   - Input shape: 1x64x64 (grayscale images)
   - Normalization: Pixel values scaled to [0, 1]

2. **Convolutional Blocks**
   - **Block 1**:
     - Conv2D: 32 filters (3x3), padding=1
     - Batch Normalization
     - ReLU activation
     - Dropout (0.2)
     - MaxPool2D (2x2)
   
   - **Block 2**:
     - Conv2D: 64 filters (3x3), padding=1
     - Batch Normalization
     - ReLU activation
     - Dropout (0.3)
     - MaxPool2D (2x2)
   
   - **Block 3**:
     - Conv2D: 128 filters (3x3), padding=1
     - Batch Normalization
     - ReLU activation
     - Dropout (0.4)
     - MaxPool2D (2x2)

3. **Residual Blocks**
   - Two residual blocks with skip connections
   - Each block contains:
     - Conv2D (3x3) → BatchNorm → ReLU → Conv2D (3x3) → BatchNorm
     - Skip connection adds input to output
     - Final ReLU activation

4. **Classification Head**
   - Global Average Pooling
   - Fully Connected Layer: 128 → 256 units
   - ReLU activation
   - Dropout (0.5)
   - Output Layer: 256 → num_classes

### Key Features
- **Residual Connections**: Help mitigate vanishing gradient problem
- **Batch Normalization**: Improves training stability
- **Progressive Dropout**: Increasing dropout rates in deeper layers
- **Global Average Pooling**: Reduces parameters before final classification

## Project Structure
```
cnn-cv-project/
├── data/                   # Dataset directory
├── models/                 # Model definitions
├── utils/                  # Utility scripts
├── config.py               # Configuration file
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Requirements
- Python 3.8+
- PyTorch or TensorFlow
- Other dependencies listed in `requirements.txt`
