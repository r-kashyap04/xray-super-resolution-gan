import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ✅ Step 1: Load the trained generator model
GENERATOR_MODEL_PATH = "generator_model.h5"

if not os.path.exists(GENERATOR_MODEL_PATH):
    raise FileNotFoundError(f"❌ ERROR: Model file '{GENERATOR_MODEL_PATH}' not found! Run train.py first.")

print("✅ Step 1: Loading trained generator model...")
generator = load_model(GENERATOR_MODEL_PATH)
print("✅ Generator model loaded successfully!")

# ✅ Step 2: Load and preprocess test images
TEST_IMAGE_PATH = "C:/Users/KANCHAN KASHYAP/Desktop/x_ray_project/chest_xray/test/"

def load_test_images(folder, target_size=(128, 128)):
    images = []
    image_names = []

    if not os.path.exists(folder) or not os.listdir(folder):
        raise ValueError(f"❌ ERROR: No test images found in '{folder}'!")

    for category in ["NORMAL", "PNEUMONIA"]:
        category_path = os.path.join(folder, category)
        if not os.path.exists(category_path):
            print(f"⚠️ WARNING: Folder '{category_path}' not found! Skipping...")
            continue
        
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale

            if img is not None:
                img_resized = cv2.resize(img, target_size)  # Resize to 128x128
                images.append(img_to_array(img_resized) / 255.0)  # Normalize (0-1)
                image_names.append(filename)
            else:
                print(f"⚠️ WARNING: Skipping unreadable image: {img_path}")

    return np.array(images), image_names

print("✅ Step 2: Loading test images...")
low_res_images, image_names = load_test_images(TEST_IMAGE_PATH)
print(f"✅ Loaded {len(low_res_images)} test images.")

if len(low_res_images) == 0:
    raise ValueError("❌ ERROR: No valid test images found!")

# ✅ Step 3: Generate high-resolution images
low_res_images = np.expand_dims(low_res_images, axis=-1)  # Add channel dimension

print("🚀 Step 3: Generating high-resolution images...")
generated_images = generator.predict(low_res_images)

generated_images = np.clip(generated_images * 255.0, 0, 255).astype(np.uint8)

print("✅ High-resolution images generated successfully!")

# ✅ Step 4: Display and save results
def display_results(low_res, high_res, image_names, num_samples=5):
    plt.figure(figsize=(10, 5))

    for i in range(min(num_samples, len(low_res))):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(low_res[i].squeeze(), cmap="gray")
        plt.title(f"Low-Res: {image_names[i]}")
        plt.axis("off")

        plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.imshow(high_res[i].squeeze(), cmap="gray")
        plt.title(f"Generated High-Res")
        plt.axis("off")

        save_path = f"generated_{image_names[i]}"
        cv2.imwrite(save_path, high_res[i].squeeze())
        print(f"📂 Saved: {save_path}")

    plt.show()

print("✅ Step 4: Displaying results...")
display_results(low_res_images, generated_images, image_names)
