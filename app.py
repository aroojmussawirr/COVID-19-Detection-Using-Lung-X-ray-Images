from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_PATH'] = os.path.join("models", "DenseNet121_model.h5")
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)
    return None

# Index
@app.route('/')
def index():
    return render_template('index.html')
# Result
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Check if DenseNet121 model exists
        if not os.path.exists(app.config['MODEL_PATH']):
            flash('DenseNet121 model not found')
            return redirect(url_for('index'))
        
        
        model = load_model(app.config['MODEL_PATH'])
        
        # Preprocess and predict
        processed_img = preprocess_image(filepath)
        if processed_img is None:
            flash('Error processing image')
            return redirect(url_for('index'))
        
        prediction = model.predict(processed_img).flatten()[0]
        
         # Interpret prediction 
        result = "COVID-19 Positive" if prediction > 0.5 else "Normal"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
           # Render results page 
        return render_template(
            'result.html',
            image_filename=filename,
            result=result,
            confidence=f"{confidence*100:.2f}%",
            model_used="DenseNet121"
        )
    
     # Handle invalid file types
    flash('Invalid file type. Only JPG, JPEG, PNG allowed.')
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the app in debug mode 
if __name__ == '__main__':
    app.run(debug=True)
