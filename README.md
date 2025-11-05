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


# Extended Model with π and e Support

## Overview
This extended model includes support for recognizing π (pi) and e symbols in addition to all standard mathematical symbols.

## Training Details

### Dataset
- **Total Images**: 10,437
- **Classes**: 20 symbols
- **Training Split**: 70% train, 15% validation, 15% test

### Symbol Distribution
| Symbol | Count | Maps To |
|--------|-------|---------|
| 0-9 | Various | Digits |
| x, y, z | Various | Variables |
| plus | 596 | + |
| minus | 655 | - |
| mul | 577 | * |
| div | 618 | / |
| dec | 624 | . |
| **pi** | **500** | **np.pi** |
| **e** | **500** | **np.e** |

### Label Mapping
The model automatically maps:
- `pi` → `np.pi` (for use with numpy/sympy)
- `e` → `np.e` (for use with numpy/sympy)

## Files Generated

### Model Files
- `equation_solver_model_extended.pt` - Trained PyTorch model
- `label_encoder_extended.pkl` - Label encoder with 20 classes
- `label_mapping_extended.pkl` - Mapping dictionary (pi→np.pi, e→np.e)
- `test_results_extended.txt` - Test accuracy and results

### Training Script
- `train_model_extended.py` - Training script that:
  - Loads all symbols from data_root except 'equals'
  - Limits π and e to 500 images each
  - Supports both .png and .jpg image formats
  - Uses data augmentation
  - Implements early stopping

### Classification Script
- `classify_symbols_extended.py` - Standalone classifier
- Can be used to test individual symbol images
- Returns mapped labels (np.pi, np.e)

## Usage

### Training
```bash
python train_model_extended.py
```

### Using in Equation Solver
The `EquationSolver` class now supports the extended model:

```python
from equation_solver import EquationSolver

# Initialize with extended model
solver = EquationSolver(use_extended_model=True)
solver.load_model()

# Now can recognize π and e in equations
# Example: 2*π = x  or  e*3 = x
```

### Web Application
The Flask app (`app.py`) has been updated to use the extended model by default:
```bash
python app.py
```

## Example Equations
The extended model can now solve equations like:
- `2*np.pi = x` → x = 6.28...
- `np.e*3 = x` → x = 8.15...
- `x + np.pi = 5` → x = 1.86...
- `np.pi * np.e = x` → x = 8.54...

## Technical Details

### Model Architecture
- **Type**: Convolutional Neural Network (CNN)
- **Layers**: 
  - 3 convolutional blocks with batch normalization
  - 2 residual blocks
  - Global average pooling
  - 2 fully connected layers
- **Input Size**: 64x64 grayscale images
- **Output**: 20 classes

### Performance
- Training uses AdamW optimizer with weight decay
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping with patience of 10 epochs
- Data augmentation during training

## Notes
- The original model (`equation_solver_model.pt`) remains unchanged
- Both models can coexist in the same directory
- Set `use_extended_model=False` to use the original model
- The extended model does NOT include 'equals' symbol (equations are split by the interface)
