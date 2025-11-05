import os
import cv2
import numpy as np
import glob
from pathlib import Path


def augment_image(img: np.ndarray, label: str) -> list:
    """
    Generate augmented versions of an image
    Returns list of (augmented_image, suffix) tuples
    """
    augmented = []
    h, w = img.shape[:2]
    
    # 1. Original
    augmented.append((img, "orig"))
    
    # 2. Small rotations (-10 to +10 degrees)
    for angle in [-8, -4, 4, 8]:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=255)
        augmented.append((rotated, f"rot{angle}"))
    
    # 3. Slight scaling (0.9x to 1.1x)
    for scale in [0.92, 1.08]:
        scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        # Pad or crop to original size
        if scale < 1:
            pad_h = (h - scaled.shape[0]) // 2
            pad_w = (w - scaled.shape[1]) // 2
            result = np.full((h, w), 255, dtype=np.uint8)
            result[pad_h:pad_h+scaled.shape[0], pad_w:pad_w+scaled.shape[1]] = scaled
        else:
            start_h = (scaled.shape[0] - h) // 2
            start_w = (scaled.shape[1] - w) // 2
            result = scaled[start_h:start_h+h, start_w:start_w+w]
        augmented.append((result, f"scale{int(scale*100)}"))
    
    # 4. Slight translations
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(img, M, (w, h), borderValue=255)
        augmented.append((shifted, f"shift{dx}_{dy}"))
    
    # 5. Slight thickness variations (for 'x', '=', operators)
    if label in ['x', 'equals', 'plus', 'minus', 'mul']:
        # Thicken
        kernel = np.ones((2, 2), np.uint8)
        thickened = cv2.erode(img, kernel, iterations=1)
        augmented.append((thickened, "thick"))
        
        # Thin
        thinned = cv2.dilate(img, kernel, iterations=1)
        augmented.append((thinned, "thin"))
    
    # 6. Slight blur (simulate different writing tools)
    blurred = cv2.GaussianBlur(img, (3, 3), 0.5)
    augmented.append((blurred, "blur"))
    
    # 7. Add slight noise
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    augmented.append((noisy, "noise"))
    
    return augmented


def augment_dataset(input_dir: str, output_dir: str, focus_classes: list = None, multiplier: int = 3):
    """
    Augment training dataset, optionally focusing on specific classes
    
    Args:
        input_dir: Original dataset directory
        output_dir: Directory to save augmented dataset
        focus_classes: List of class labels to augment more (e.g., ['x', 'equals'])
        multiplier: How many times more to augment focus classes
    """
    print(f"{'='*60}")
    print("Augmenting Dataset")
    print(f"{'='*60}")
    
    if focus_classes is None:
        focus_classes = ['x', 'equals']
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in Path(input_dir).iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        label = class_dir.name
        output_class_dir = Path(output_dir) / label
        output_class_dir.mkdir(exist_ok=True)
        
        # Determine if this is a focus class
        is_focus = label in focus_classes
        repeat_count = multiplier if is_focus else 1
        
        # Get all images in this class
        image_files = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
        
        print(f"\nProcessing class '{label}' ({len(image_files)} images)")
        if is_focus:
            print(f"  [FOCUS CLASS - {multiplier}x augmentation]")
        
        total_generated = 0
        
        for img_path in image_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Generate augmented versions
            augmented_list = augment_image(img, label)
            
            # For focus classes, use all augmentations; for others, use subset
            if is_focus:
                selected_aug = augmented_list
            else:
                # Use original + a few basic augmentations
                selected_aug = augmented_list[:5]
            
            # Save augmented images
            base_name = img_path.stem
            for repeat_idx in range(repeat_count):
                for aug_img, suffix in selected_aug:
                    output_name = f"{base_name}_{suffix}_r{repeat_idx}.png"
                    output_path = output_class_dir / output_name
                    cv2.imwrite(str(output_path), aug_img)
                    total_generated += 1
        
        print(f"  Generated {total_generated} images for class '{label}'")
    
    print(f"\n{'='*60}")
    print(f"Augmentation complete!")
    print(f"Augmented dataset saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Augment training dataset')
    parser.add_argument('--input', required=True, help='Input dataset directory')
    parser.add_argument('--output', required=True, help='Output directory for augmented dataset')
    parser.add_argument('--focus', nargs='+', default=['x', 'equals'], 
                       help='Classes to augment more (default: x equals)')
    parser.add_argument('--multiplier', type=int, default=3,
                       help='How many times more to augment focus classes (default: 3)')
    
    args = parser.parse_args()
    
    augment_dataset(args.input, args.output, args.focus, args.multiplier)
