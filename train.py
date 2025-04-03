import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.optimizers import Adam
import os
import numpy as np

# Constants
INPUT_SIZE = (299, 299)  # InceptionV3 default input size
TEST_MODE = False  # Set to False for full training


def test_with_single_image():
    """Test the model pipeline with a single image"""
    # Load a test image (replace with your image path)
    test_img_path = 'data/test/Normal/image1.jpg'  # or 'data/test/TB/example.jpg'
    try:
        img = load_img(test_img_path, target_size=INPUT_SIZE)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Create a dummy model
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(*INPUT_SIZE, 3))
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(1, activation='sigmoid')
        ])

        # Test prediction
        prediction = model.predict(img_array)
        print(f"\nSingle image test prediction: {prediction[0][0]}")
        print("Test successful! Pipeline works with single image.")
        return True

    except Exception as e:
        print(f"\nError in single image test: {str(e)}")
        return False


def create_data_generators():
    """Create data generators with enhanced augmentation"""
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        'data/train/',
        target_size=INPUT_SIZE,
        batch_size=32,
        class_mode='binary',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        'data/test/',
        target_size=INPUT_SIZE,
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, val_generator


def build_and_train_model():
    """Build and train the model"""
    train_gen, val_gen = create_data_generators()

    base_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(*INPUT_SIZE, 3)
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    print("\nTraining model...")
    history = model.fit(
        train_gen,
        epochs=2 if TEST_MODE else 10,
        validation_data=val_gen
    )

    # Save model
    model.save('model/tb_model.h5')
    print("Model saved successfully")

    return model, history


def convert_to_tflite(model):
    """Convert the model to TFLite format"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    os.makedirs('model', exist_ok=True)
    with open('model/tb_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Model converted to TFLite")


if __name__ == "__main__":
    # First test with single image
    if TEST_MODE:
        print("Running in TEST MODE (single image check)")
        if not test_with_single_image():
            exit()

    # Then proceed with training if test passes
    try:
        model, history = build_and_train_model()
        convert_to_tflite(model)
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
