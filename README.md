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
