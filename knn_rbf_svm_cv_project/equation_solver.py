import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sympy as sp
from cv_preprocess import preprocess_image
from segmentation import extract_rois
from recognition import Recognizer
from parser_ast import build_sequence, parse_to_sympy


class HandwrittenEquationSolver:
    def __init__(self, models_dir: str = 'models'):
        self.recognizer = Recognizer(models_dir=models_dir, prefer='svm')

    def process_equation(self, image_path: str):
        data = preprocess_image(image_path)
        gray = data['gray']
        bin_img = data['binary']
        rois = extract_rois(gray, bin_img)
        if not rois:
            return "No characters found in the image.", None

        symbols = []
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for r in rois:
            ch = self.recognizer.predict_one(r['roi'])
            symbols.append({'char': ch, 'is_superscript': r['is_superscript']})
            x, y, w, h = r['bbox']
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis, ch, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        seq = build_sequence(symbols)
        expr = parse_to_sympy(seq)
        if expr is None:
            return f"Could not parse expression: {seq}", vis

        try:
            if isinstance(expr, sp.Equality):
                sol = sp.solve(expr)
                return f"Solution: {sol}", vis
            else:
                val = sp.simplify(expr)
                return f"Result: {val}", vis
        except Exception as e:
            return f"Error evaluating expression: {e}", vis


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default='handwritten_equation.png', help='Path to input image')
    ap.add_argument('--models', default='models', help='Path to trained models directory')
    args = ap.parse_args()

    solver = HandwrittenEquationSolver(models_dir=args.models)
    result, vis = solver.process_equation(args.image)
    print(result)
    if vis is not None:
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title("Recognized and Parsed")
        plt.axis('off')
        plt.show()