import cv2
import numpy as np
from PIL import Image

def extract_frames(video_path, num_frames=30):
    """
    Extract frames from video
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    # Calculate frame interval
    interval = max(total_frames // num_frames, 1)
    
    for i in range(num_frames):
        frame_idx = i * interval
        if frame_idx >= total_frames:
            break
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return frames

def preprocess_video_frames(frames, target_size=(224, 224)):
    """
    Preprocess video frames for model input
    """
    processed_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        processed_frames.append(img_array)
    
    # Pad with zeros if we don't have enough frames
    while len(processed_frames) < 30:
        processed_frames.append(np.zeros((*target_size, 3)))
    
    return np.array(processed_frames[:30])  # Ensure we return exactly 30 frames