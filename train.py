import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from preprocess import load_data  # Ensure load_data returns (high_res, low_res)
from generator import build_generator
from discriminator import build_discriminator
from srgan import build_srgan

# ✅ Set Paths
DATASET_PATH = "C:/Users/KANCHAN KASHYAP/Desktop/x_ray_project/chest_xray/train"
EPOCHS = 500

BATCH_SIZE = 16

# ✅ Load Dataset (Ensure it returns high & low resolution images)
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"❌ Dataset path '{DATASET_PATH}' not found!")

print("✅ Loading dataset...")
high_res_images, low_res_images = load_data(DATASET_PATH)  # Fix dataset loading

# ✅ Build or Load Models
print("✅ Loading/Building Models...")
generator = build_generator()
discriminator = build_discriminator()

# ✅ Compile Discriminator
discriminator.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5), metrics=["accuracy"])

# ✅ Build and Compile SRGAN
srgan = build_srgan(generator, discriminator)
srgan.compile(loss=['mse', 'binary_crossentropy'], optimizer=Adam(0.0002, 0.5))

# ✅ Training Loop
print("🚀 Starting Training...")
for epoch in range(EPOCHS):
    # ✅ Select a batch of real & low-res images
    idx = np.random.randint(0, high_res_images.shape[0], BATCH_SIZE)
    real_batch = high_res_images[idx]
    low_res_batch = low_res_images[idx]  # Low-res images for generator input

    # ✅ Generate a batch of fake images (SR output)
    fake_images = generator.predict(low_res_batch)

    # ✅ Create labels for real and fake images
    real_labels = np.ones((BATCH_SIZE, 1))
    fake_labels = np.zeros((BATCH_SIZE, 1))

    # ✅ Train Discriminator
    d_loss_real = discriminator.train_on_batch(real_batch, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)  # Average real & fake loss

    # ✅ Train Generator (SRGAN)
    misleading_labels = np.ones((BATCH_SIZE, 1))  # Trick discriminator
    g_loss = srgan.train_on_batch(low_res_batch, [real_batch, misleading_labels])

    # ✅ Print progress every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: D Loss = {d_loss[0]:.4f}, G Loss = {g_loss[0]:.4f}")

# ✅ Save Trained Models
generator.save("generator_model.keras")
discriminator.save("discriminator_model.keras")
srgan.save("srgan_model.keras")

print("✅ Training Completed! Models Saved Successfully!")
