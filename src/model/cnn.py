import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(num_classes: int, input_shape=(224, 224, 3)):
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )

    inputs = layers.Input(shape=input_shape, name="image")

    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)

    for filters, dropout in [(32, 0.10), (64, 0.15), (128, 0.20), (256, 0.25)]:
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(dropout)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    return models.Model(inputs, outputs, name="plant_classifier")
