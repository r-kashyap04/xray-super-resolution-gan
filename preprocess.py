import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array

def load_data(dataset_path, target_size=(256, 256)):
    high_res_images = []
    low_res_images = []
    
    for subfolder in ["NORMAL", "PNEUMONIA"]:
        subfolder_path = os.path.join(dataset_path, subfolder)
        for filename in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            if img is not None:
                img_resized = cv2.resize(img, target_size)  # Resize to 256x256
                high_res_images.append(img_to_array(img_resized) / 255.0)  # Normalize
                low_res_images.append(cv2.resize(img_resized, (128, 128)))  # Downsample to 128x128
    
    return np.array(high_res_images), np.array(low_res_images)

# Set dataset path (Update the path as needed)
dataset_path = "C:/Users/KANCHAN KASHYAP/Desktop/x_ray_project/chest_xray/train/"
high_res_images, low_res_images = load_data(dataset_path)

# Print dataset info
print(f"✅ Loaded {len(high_res_images)} high-res and {len(low_res_images)} low-res images")

# Display a sample image
plt.figure(figsize=(10, 5))

# Show low-resolution image
plt.subplot(1, 2, 1)
plt.imshow(low_res_images[0], cmap='gray')
plt.title("Low-Resolution (128x128)")

# Show high-resolution image
plt.subplot(1, 2, 2)
plt.imshow(high_res_images[0], cmap='gray')
plt.title("High-Resolution (256x256)")

plt.show()
