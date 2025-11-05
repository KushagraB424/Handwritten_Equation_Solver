import cv2
import numpy as np
from equation_solver import EquationSolver
import os
from pathlib import Path

def create_test_equation():
    """Create a test equation image by combining individual symbols"""
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create a blank image (white background)
    eq_height = 200
    eq_width = 800
    equation_img = np.ones((eq_height, eq_width), dtype=np.uint8) * 255
    
    # Load and place symbols (4x+3=10)
    symbols_to_load = ['4', 'x', 'plus', '3', 'equals', '1', '0']
    x_offset = 50  # Starting position
    
    data_root = Path("data_root")
    for symbol in symbols_to_load:
        # Get first image from symbol's directory
        symbol_dir = data_root / str(symbol)
        if symbol_dir.exists():
            # Try PNG files first, then JPG
            image_files = list(symbol_dir.glob("*.png"))
            if not image_files:
                image_files = list(symbol_dir.glob("*.jpg"))
            
            for img_path in image_files:
                symbol_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if symbol_img is None:
                    continue
                
                # Ensure black text on white background
                if np.mean(symbol_img) < 127:
                    symbol_img = 255 - symbol_img
                
                # Enhance contrast
                symbol_img = cv2.equalizeHist(symbol_img)
                _, symbol_img = cv2.threshold(symbol_img, 127, 255, cv2.THRESH_BINARY)
                
                # Resize while maintaining aspect ratio
                h, w = symbol_img.shape
                new_h = 100  # Fixed height
                new_w = int(w * (new_h / h))
                if new_w < 10:  # Skip if too narrow
                    continue
                symbol_img = cv2.resize(symbol_img, (new_w, new_h))
                
                # Add padding around symbol
                pad = 20
                padded_symbol = np.ones((new_h + 2*pad, new_w + 2*pad), dtype=np.uint8) * 255
                padded_symbol[pad:-pad, pad:-pad] = symbol_img
                
                # Place symbol in equation image
                h, w = padded_symbol.shape
                y_offset = (eq_height - h) // 2
                if x_offset + w < eq_width:
                    equation_img[y_offset:y_offset+h, x_offset:x_offset+w] = padded_symbol
                    x_offset += w + 30  # Add more spacing between symbols
                break
    
    # Save the test equation
    test_path = test_dir / "test_equation.png"
    cv2.imwrite(str(test_path), equation_img)
    
    # Also save a debug version showing boundaries
    debug_img = cv2.cvtColor(equation_img, cv2.COLOR_GRAY2BGR)
    _, binary = cv2.threshold(equation_img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(debug_img, contours, -1, (0,255,0), 2)
    cv2.imwrite(str(test_dir / "test_equation_debug.png"), debug_img)
    
    return str(test_path)

def visualize_segments(img, symbols):
    """Visualize segmented symbols"""
    # Create a color version of the input image
    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw bounding boxes around detected symbols
    for i, symbol in enumerate(symbols):
        x = i * (symbol.shape[1] + 10)  # Add spacing between symbols
        y = img.shape[0] + 10
        h, w = symbol.shape
        
        # Draw the symbol below the equation
        canvas_height = img.shape[0] + h + 20
        if i == 0:
            # Create new canvas with space for symbols below
            canvas = np.ones((canvas_height, img.shape[1], 3), dtype=np.uint8) * 255
            canvas[:img.shape[0], :img.shape[1]] = vis_img
            vis_img = canvas
        
        # Place segmented symbol
        if x + w < vis_img.shape[1]:
            vis_img[y:y+h, x:x+w] = cv2.cvtColor(symbol, cv2.COLOR_GRAY2BGR)
    
    return vis_img

def test_training():
    """Test model training and symbol classification"""
    solver = EquationSolver()
    solver.train_classifier()
    solver.save_model()
    print("Model trained and saved successfully!")

def test_equation(equation_img_path: str):
    """Test equation parsing and solving"""
    solver = EquationSolver()
    try:
        solver.load_model()
    except FileNotFoundError:
        print("Training new model first...")
        solver.train_classifier()
        solver.save_model()
    
    # Load and process test equation
    img = cv2.imread(equation_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {equation_img_path}")
    
    # First show the input image
    cv2.imshow("Input Equation", img)
    cv2.waitKey(1000)
    
    # Get segmented symbols and visualize
    symbols = solver.segment_equation(img)
    vis_img = visualize_segments(img, symbols)
    cv2.imshow("Segmented Symbols", vis_img)
    cv2.waitKey(1000)
    
    # Solve equation
    try:
        solution = solver.solve_equation(img)
        print(f"Solution: {solution}")
        
        # Display result
        result_img = cv2.putText(vis_img.copy(), f"Solution: {solution}", 
                               (10, vis_img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 0, 255), 2)
        cv2.imshow("Result", result_img)
        print("Press any key in the image window to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Done!")
    except Exception as e:
        print(f"Error solving equation: {e}")

if __name__ == "__main__":
    # Test with the provided image
    test_img_path = "testthis.png"
    print(f"Testing with equation image: {test_img_path}")
    test_equation(test_img_path)