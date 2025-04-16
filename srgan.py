import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from generator import build_generator
from discriminator import build_discriminator

def build_srgan(generator, discriminator):
    # Define input layer
    input_layer = Input(shape=(128, 128, 1))
    
    # Generate high-resolution images
    generated_image = generator(input_layer)
    
    # Ensure discriminator does not update while training generator
    discriminator.trainable = False
    
    # Classify the generated image
    validity = discriminator(generated_image)
    
    # Define the SRGAN model
    srgan_model = Model(inputs=input_layer, outputs=[generated_image, validity])
    
    return srgan_model

if __name__ == "__main__":
    # ✅ Load generator and discriminator models
    generator = build_generator()
    discriminator = build_discriminator()
    
    # ✅ Build and compile SRGAN
    srgan = build_srgan(generator, discriminator)
    
    srgan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                  loss=['mse', 'binary_crossentropy'])
    
    srgan.summary()
    srgan.save("srgan_model.h5")  # ✅ Save model correctly
    print("✅ SRGAN Model Created and Saved Successfully!")
