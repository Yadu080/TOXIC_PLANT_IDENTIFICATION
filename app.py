"""
app.py – Flask backend for Toxic‑Plant Identification
----------------------------------------------------
✓ defines upload folder correctly
✓ makes sure the folder exists
✓ uses a single Flask app instance
✓ uses correct template names
✓ joins paths with app.config['UPLOAD_FOLDER']
"""
from werkzeug.utils import secure_filename   # add this at the top with other imports
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import cv2
import os
import shutil

# ──────────────────── 1.  Flask app & folders ────────────────────
app = Flask(__name__)

# Where uploaded files will be stored (relative to project root)
UPLOAD_FOLDER = os.path.join('static', 'uploads')  # → static/uploads
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ──────────────────── 2.  Load the trained model ──────────────────
model = tf.keras.models.load_model('toxic_plant_classifier.h5')

# ──────────────────── 3.  Helper functions ───────────────────────

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def classify_plant(image_path: str, model, image_size=(128, 128)):
    """
    Returns ('Toxic'|'Non‑Toxic', probability_of_toxic) just like the notebook.
    """
    img = cv2.imread(image_path)                 # BGR
    if img is None:
        raise FileNotFoundError(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # <─ convert to RGB
    img = cv2.resize(img, image_size)
    img = np.expand_dims(img / 255.0, 0)

    prob = float(model.predict(img)[0][0])       # sigmoid → scalar
    label = "Toxic" if prob > 0.5 else "Non‑Toxic"
    return label, prob

# ──────────────────── Routes ──────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    # 1. make sure a file was posted
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('home'))

    # 2. save the file → static/uploads/<filename>
    filename = secure_filename(file.filename)            # sanitize
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # 3. classify the saved image
    label, score = classify_plant(save_path, model)

    # 4. render result page
    return render_template(
        'result.html',
        classification=label,
        confidence=score,      # float between 0‑1
        filename=filename      # used by result.html to show the image
    )

# ──────────────────── 5.  (Optional) test route ───────────────────
@app.route('/test')
def test():
    return "<h1>Test Page</h1>"

# ──────────────────── 6.  Main entry point ────────────────────────
if __name__ == '__main__':
    # change port if 5000 is busy:  app.run(debug=True, port=5001)
    app.run(debug=True)
