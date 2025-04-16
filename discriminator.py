import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Model, Sequential

def build_discriminator(input_shape=(256, 256, 1)):
    model = Sequential([
        Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=input_shape),
        LeakyReLU(alpha=0.2),

        Conv2D(128, kernel_size=4, strides=2, padding="same"),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Conv2D(256, kernel_size=4, strides=2, padding="same"),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    
    return model

if __name__ == "__main__":
    discriminator = build_discriminator()
    discriminator.summary()
    print("✅ Discriminator Model Created Successfully!")
