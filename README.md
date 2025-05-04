# Deepfake Detection Using Deep Learning

![Project Banner](frontend/public/img1.jpg)

A web-based application that detects deepfake images and videos using deep learning models, providing real-time analysis with confidence scores.

## Features

- 🖼️ **Image Analysis**: Detect deepfakes in single images
- 🎥 **Video Analysis**: Analyze videos frame-by-frame for deepfake detection
- 📊 **Confidence Scores**: Get percentage-based predictions
- 🚀 **User-Friendly Interface**: Simple web interface for easy usage
- ⚡ **Fast Processing**: Optimized models for quick analysis

## Technologies Used

### Frontend
- React.js
- Axios (for API calls)
- CSS3

### Backend
- Flask (Python web framework)
- TensorFlow/Keras (Deep Learning)
- OpenCV (Video processing)

### Models
- **Image Detection**: EfficientNetB0-based CNN
- **Video Detection**: 3D CNN with temporal analysis

## Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- TensorFlow 2.6+
- GPU with CUDA support (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/deepfake-detection.git
   cd deepfake-detection
   ```
2. Backend Setup
   ```
   cd backend
   pip install -r requirements.txt
   ```
3. Frontend Setup
   ```
   cd ../frontend
   npm install
   ```
4. Download Pretrained Models
   - Place models in backend/models/ directory
   - Or train your own (see Training section below)
  
### Usage

1. Start Backend Server
   ```
   cd backend
   python app.py
   ```
2. Start Frontend Development Server
   ```
   cd ../frontend
   npm start
   ```
3. Access the Application
   - Open http://localhost:3000 in your browser
   - Upload an image or video for analysis

### Training Models
## Image Model Training
   ```
   python model_training/train_image_model.py
   ```
## Video Model Training
   ```
   python model_training/train_video_model.py
   ```
### Results
   - Real Image: 92% confidence
   - Fake Image: 87% confidence
   - Video Analysis: 30 frames processed in 4.2s
     
### Contributing
   - Fork the project
   - Create your feature branch (git checkout -b feature/AmazingFeature)
   - Commit your changes (git commit -m 'Add some amazing feature')
   - Push to the branch (git push origin feature/AmazingFeature)
   - Open a Pull Request
   
