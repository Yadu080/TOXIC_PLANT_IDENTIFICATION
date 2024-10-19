from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import numpy as np
import cv2
import os
import shutil

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('toxic_plant_classifier.h5')

# Set allowed file extensions for images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to classify the uploaded plant image
def classify_plant(image_path, model, image_size=(128, 128)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_size)
    img = np.expand_dims(img / 255.0, axis=0)  # Preprocessing
    prediction = model.predict(img)[0][0]
    return 'Toxic' if prediction > 0.5 else 'Non-Toxic'

# Homepage route (upload form)
@app.route('/')
def home():
    return render_template('templates/index.html')

# Route for handling the image upload and classification
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        
        # Classify the uploaded image
        classification = classify_plant(filepath, model)
        
        # Copy the file to static/uploads for display
        shutil.copy(filepath, os.path.join('static/uploads', filename))
        
        # Return the result to the frontend
        return render_template('templates/result.html', classification=classification, filename=filename)
    
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    print("Rendering homepage...")
    return render_template('index.html')

# Add a new test route to confirm Flask is serving any HTML
@app.route('/test')
def test():
    return "<h1>Test Page</h1>"

if __name__ == "__main__":
    app.run(debug=True)
