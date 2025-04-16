import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Add, UpSampling2D, Input
from tensorflow.keras.models import Model

def residual_block(input_layer):
    x = Conv2D(64, kernel_size=3, padding="same")(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    
    # ✅ Fix Add Layer by passing inputs as a list
    x = Add()([input_layer, x])
    
    return x

def build_generator():
    input_layer = Input(shape=(128, 128, 1))
    
    x = Conv2D(64, kernel_size=3, padding="same")(input_layer)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Creating multiple residual blocks
    for _ in range(8):  
        x = residual_block(x)
    
    x = UpSampling2D(size=2)(x)
    x = Conv2D(64, kernel_size=3, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(1, kernel_size=3, padding="same", activation="tanh")(x)
    
    model = Model(inputs=input_layer, outputs=x)
    return model

if __name__ == "__main__":
    generator = build_generator()
    generator.summary()
    generator.save("generator_model.h5")  # ✅ Save model correctly
    print("✅ Generator Model Created and Saved Successfully!")
