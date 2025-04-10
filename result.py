import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import os

# Constants
INPUT_SIZE = (299, 299)  # InceptionV3 default input size
BATCH_SIZE = 32
EPOCHS = 10


def create_data_generators():
    """Create data generators with validation split"""
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )

    train_generator = train_datagen.flow_from_directory(
        'data/train/',
        target_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        'data/train/',
        target_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, val_generator


def build_model():
    """Build InceptionV3 based model"""
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
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model


def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()


def generate_classification_report(model, generator):
    """Generate and save classification metrics"""
    y_true = generator.classes
    y_pred = (model.predict(generator) > 0.5).astype(int)

    # Classification report
    report = classification_report(y_true, y_pred,
                                   target_names=generator.class_indices.keys())
    with open('classification_report.txt', 'w') as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=generator.class_indices.keys(),
                yticklabels=generator.class_indices.keys())
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

    return report


def main():
    # Create data generators
    train_gen, val_gen = create_data_generators()

    # Build and train model
    model = build_model()
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen
    )

    # Generate visualizations and reports
    plot_training_history(history)

    # Convert the continuous probabilities to binary labels
    val_pred_probs = model.predict(val_gen)
    val_pred_labels = (val_pred_probs > 0.5).astype(int)  # For binary classification

    # Generate classification report
    report_data = classification_report(val_gen.classes, val_pred_labels, output_dict=True)

    # Print dataset and model info
    print("\n=== Dataset Information ===")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print("Class distribution:")
    for cls, idx in train_gen.class_indices.items():
        print(f"  {cls}: {np.sum(train_gen.classes == idx)} samples")

    print("\n=== Classification Report ===")
    print(report_data)

    print("\n=== Final Metrics ===")
    print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.2f}")
    print(f"Validation Precision: {history.history['val_precision'][-1]:.2f}")
    print(f"Validation Recall: {history.history['val_recall'][-1]:.2f}")

    # Prepare data for CSV report
    final_metrics = {
        "Validation Accuracy": history.history['val_accuracy'][-1],
        "Validation Precision": history.history['val_precision'][-1],
        "Validation Recall": history.history['val_recall'][-1],
    }

    class_distribution = {cls: np.sum(train_gen.classes == idx) for cls, idx in train_gen.class_indices.items()}

    # Create a DataFrame for the final metrics and class distribution
    metrics_df = pd.DataFrame([final_metrics])
    class_distribution_df = pd.DataFrame(list(class_distribution.items()), columns=["Class", "Samples"])

    # Ensure the 'report' folder exists
    os.makedirs('reports', exist_ok=True)

    # Save results to CSV inside the 'report' folder
    metrics_df.to_csv('reports/training_metrics_report.csv', index=False)
    class_distribution_df.to_csv('reports/class_distribution_report.csv', index=False)

    # Save classification report to CSV inside the 'report' folder
    report_df = pd.DataFrame(report_data).transpose()
    report_df.to_csv('reports/classification_report.csv', index=True)

    print("\n=== Reports saved to CSV files inside the 'reports' folder ===")


if __name__ == "__main__":
    main()
