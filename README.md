# Deepfake Detection Using Deep Learning

![Project Banner](frontend/public/img1.jpg) *(Optional: Add your banner image here)*

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