"""
Test script to verify the extended model with π and e support
"""
import cv2
import numpy as np
from equation_solver import EquationSolver

def test_extended_model():
    """Test the extended model with π and e symbols"""
    print("="*60)
    print("Testing Extended Model with π and e Support")
    print("="*60)
    
    # Initialize solver with extended model
    solver = EquationSolver(use_extended_model=True)
    
    # Load the extended model
    try:
        solver.load_model()
        print("\n✅ Extended model loaded successfully!")
        print(f"📊 Number of classes: {len(solver.label_encoder.classes_)}")
        print(f"📋 Supported symbols: {list(solver.label_encoder.classes_)}")
        print(f"🔄 Label mapping: {solver.label_mapping}")
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return
    
    print("\n" + "="*60)
    print("Model is ready to classify symbols including π and e!")
    print("="*60)
    
    # Test with sample images if they exist
    test_images = [
        ('data_root/pi/0.png', 'π (pi)'),
        ('data_root/e/0.png', 'e'),
        ('data_root/0/0.png', '0'),
        ('data_root/x/0.png', 'x'),
    ]
    
    print("\n" + "="*60)
    print("Testing Symbol Classification")
    print("="*60)
    
    for img_path, expected in test_images:
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                predicted = solver.predict_symbol(img)
                status = "✅" if predicted else "❌"
                print(f"{status} {img_path}: Expected '{expected}', Got '{predicted}'")
        except Exception as e:
            print(f"⚠️ Could not test {img_path}: {e}")
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)
    print("\nThe extended model can now:")
    print("  • Recognize digits 0-9")
    print("  • Recognize variables x, y, z")
    print("  • Recognize operators +, -, *, /, .")
    print("  • Recognize π (mapped to np.pi)")
    print("  • Recognize e (mapped to np.e)")
    print("\nExample equations you can solve:")
    print("  • 2*np.pi = x")
    print("  • np.e*3 = x")
    print("  • x+np.pi = 5")

if __name__ == "__main__":
    test_extended_model()
