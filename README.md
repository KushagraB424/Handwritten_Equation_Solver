# Handwritten Equation Solver

A Python application that uses classical computer vision and machine learning to recognize and solve handwritten mathematical equations.

## Features

- Preprocesses images of handwritten equations
- Segments and recognizes individual digits and operators
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

1. **Image Preprocessing**: Converts the image to grayscale and applies adaptive thresholding.
2. **Contour Detection**: Finds and extracts individual characters in the equation.
3. **Character Recognition**: Uses a pre-trained model to recognize digits and operators.
4. **Equation Solving**: Parses the recognized characters and solves the equation.

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- scikit-image
- TensorFlow
- SymPy
- Matplotlib

## Note

For best results, use clear handwriting and ensure good contrast between the text and background. The current model is a simple example and may need to be trained on a larger dataset for better accuracy.


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

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
ended_model=False` to use the original model
- The extended model does NOT include 'equals' symbol (equations are split by the interface)
