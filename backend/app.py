from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
from utils.image_processing import preprocess_image
from utils.video_processing import extract_frames, preprocess_video_frames

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
IMAGE_MODEL = load_model('models/image_model.h5')
VIDEO_MODEL = load_model('models/video_model.h5')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Determine if it's an image or video
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        if file_ext in {'png', 'jpg', 'jpeg'}:
            # Process image
            img = Image.open(filepath)
            processed_img = preprocess_image(img)
            prediction = IMAGE_MODEL.predict(np.expand_dims(processed_img, axis=0))
            result = 'Fake' if prediction[0][0] > 0.5 else 'Real'
            confidence = float(prediction[0][0]) if result == 'Fake' else 1 - float(prediction[0][0])
            
            return jsonify({
                'type': 'image',
                'result': result,
                'confidence': round(confidence * 100, 2),
                'filename': filename
            })
            
        elif file_ext in {'mp4', 'mov'}:
            # Process video
            frames = extract_frames(filepath, num_frames=30)
            processed_frames = preprocess_video_frames(frames)
            prediction = VIDEO_MODEL.predict(np.expand_dims(processed_frames, axis=0))
            result = 'Fake' if prediction[0][0] > 0.5 else 'Real'
            confidence = float(prediction[0][0]) if result == 'Fake' else 1 - float(prediction[0][0])
            
            return jsonify({
                'type': 'video',
                'result': result,
                'confidence': round(confidence * 100, 2),
                'filename': filename
            })
    
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)