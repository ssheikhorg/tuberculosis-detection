import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os


def create_data_generators():
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation data generator (only rescaling)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    # Flow from directory
    train_generator = train_datagen.flow_from_directory(
        'data/train/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        'data/test/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    return train_generator, val_generator


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def convert_to_tflite(model):
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the TFLite model
    os.makedirs('model', exist_ok=True)
    with open('model/tb_model.tflite', 'wb') as f:
        f.write(tflite_model)

    print("Model converted to TFLite and saved")


def train_and_save_model():
    # Create data generators
    train_gen, val_gen = create_data_generators()

    # Build and compile model
    model = build_model()

    # Train the model
    print("Training model...")
    model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen
    )

    # Save Keras model (optional)
    model.save('model/tb_model.h5')

    # Convert to TFLite
    convert_to_tflite(model)


if __name__ == "__main__":
    train_and_save_model()