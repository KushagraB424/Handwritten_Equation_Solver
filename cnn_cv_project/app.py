from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import io
import os
from sympy import symbols, Eq, solve, sympify
from equation_solver import EquationSolver

# Flask setup
app = Flask(__name__)
# Use extended model with π and e support
solver = EquationSolver(use_extended_model=True)

# Load model once at startup
try:
    solver.load_model()
    print("✅ Extended model loaded successfully!")
    print(f"📊 Supported symbols: {solver.label_encoder.classes_}")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")


# -----------------------------
# Helper: Decode base64 → OpenCV image
# -----------------------------
def decode_image(base64_string):
    if 'base64,' in base64_string:
        base64_string = base64_string.split('base64,')[1]
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return img


# -----------------------------
# Helper: Create plot for equation and solution
# -----------------------------
def format_solution(solution):
    """Format the solution dictionary into a readable string."""
    if not isinstance(solution, dict):
        return str(solution)
    
    parts = []
    for var, val in solution.items():
        if isinstance(val, (int, float)):
            # Format numbers nicely (e.g., 2.0 becomes 2, 2.5 stays as is)
            if val.is_integer():
                parts.append(f"{var} = {int(val)}")
            else:
                parts.append(f"{var} = {val:.2f}".rstrip('0').rstrip('.'))
        else:
            parts.append(f"{var} = {val}")
    
    return ", ".join(parts)



# -----------------------------
# Route: Home Page
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')


# -----------------------------
# Route: Solve Equation
# -----------------------------
@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.get_json(force=True)
        left_img_data = data.get('leftImage')
        right_img_data = data.get('rightImage')

        if not left_img_data or not right_img_data:
            return jsonify({'success': False, 'error': 'Missing left or right image data'})

        # Decode both sides
        left_img = decode_image(left_img_data)
        right_img = decode_image(right_img_data)

        # Optional: Save for debugging
        cv2.imwrite('debug_left.png', left_img)
        cv2.imwrite('debug_right.png', right_img)

        # -----------------------------
        # Recognize and Solve using EquationSolver helpers
        # -----------------------------
        try:
            solved = solver.solve_from_sides(left_img, right_img)
        except ValueError as err:
            return jsonify({'success': False, 'error': str(err)})

        equation_str = solved['equation']
        solution = solved['solution']

        print(f"🧮 Parsed equation: {equation_str}")
        print(f"🔍 Solution: {solution}")
        print(f"📝 Left expression: {solved.get('left')}")
        print(f"📝 Right expression: {solved.get('right')}")

        # Format the solution for display
        formatted_solution = format_solution(solution)
        print(f"🔍 Formatted solution: {formatted_solution}")

        return jsonify({
            'success': True,
            'equation': equation_str,
            'solution': solution,
            'formatted_solution': formatted_solution,
            'leftExpression': solved.get('left', ''),
            'rightExpression': solved.get('right', '')
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
