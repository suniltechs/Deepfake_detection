import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Constants
FRAME_SIZE = (224, 224)
SEQUENCE_LENGTH = 30
BATCH_SIZE = 8
EPOCHS = 25
NUM_CLASSES = 2  # Real and Fake

# Data preparation
def load_video_frames(video_path, max_frames=SEQUENCE_LENGTH):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    try:
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, FRAME_SIZE)
            frame = frame / 255.0  # Normalize
            frames.append(frame)
            
        # Pad with black frames if needed
        while len(frames) < max_frames:
            frames.append(np.zeros((*FRAME_SIZE, 3)))
            
    finally:
        cap.release()
        
    return np.array(frames)

# Example dataset structure: {'real': [video_paths], 'fake': [video_paths]}
def prepare_dataset(data_dir):
    real_videos = [os.path.join(data_dir, 'real', f) for f in os.listdir(os.path.join(data_dir, 'real'))]
    fake_videos = [os.path.join(data_dir, 'fake', f) for f in os.listdir(os.path.join(data_dir, 'fake'))]
    
    X = []
    y = []
    
    for video in real_videos:
        X.append(load_video_frames(video))
        y.append(0)  # 0 for real
        
    for video in fake_videos:
        X.append(load_video_frames(video))
        y.append(1)  # 1 for fake
        
    return np.array(X), np.array(y)

# Model architecture (3D CNN)
def create_video_model():
    model = models.Sequential([
        layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(SEQUENCE_LENGTH, *FRAME_SIZE, 3)),
        layers.MaxPooling3D((2, 2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv3D(64, (3, 3, 3), activation='relu'),
        layers.MaxPooling3D((2, 2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv3D(128, (3, 3, 3), activation='relu'),
        layers.MaxPooling3D((2, 2, 2)),
        layers.BatchNormalization(),
        
        layers.GlobalAveragePooling3D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model

# Prepare dataset
X, y = prepare_dataset('dataset/videos')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
video_model = create_video_model()

# Callbacks
callbacks = [
    ModelCheckpoint(
        'models/video_model.h5',
        save_best_only=True,
        monitor='val_accuracy'
    ),
    EarlyStopping(
        patience=5,
        restore_best_weights=True
    )
]

# Training
history = video_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('video_training_accuracy.png')
plt.show()