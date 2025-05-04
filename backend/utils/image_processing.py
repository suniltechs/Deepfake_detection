import numpy as np
from PIL import Image

def preprocess_image(img, target_size=(224, 224)):
    """
    Preprocess image for model input
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array