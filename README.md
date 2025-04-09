# Image Classification Model for Tuberculosis Detection Using InceptionV3 and TensorFlow Lite

## Abstract
This project uses deep learning models to develop an AI-driven system for detecting Tuberculosis (TB) from chest X-ray images. By utilizing a pre-trained InceptionV3 model for feature extraction and building a custom classifier on top, this system aims to improve classification accuracy and efficiency. The model is trained and fine-tuned on chest X-ray data, converted to TensorFlow Lite for mobile compatibility, and deployed using FastAPI to provide real-time predictions. This report describes the methodology used to enhance the model's performance, evaluate its results, and deploy it for practical use.

#### Data Augmentation Techniques:
- Rotation (up to 30 degrees)
- Width and height shifts (up to 20%)
- Shear and zoom transformations
- Horizontal and vertical flips
- Brightness adjustments

This augmentation helps ensure the model generalizes well to various transformations of the images.

### Model Development
The model is based on the InceptionV3 architecture, which has been pre-trained on the ImageNet dataset. 

### Deployment Using FastAPI
The trained model is deployed via FastAPI, a web framework for building APIs with Python. The FastAPI application allows users to upload chest X-ray images, which are then processed and classified by the model. The prediction result is returned as a JSON response, indicating whether the image is classified as "TB Positive" or "Normal."

#### FastAPI Features:
- Endpoint to check the health of the application and model.
- Template-based HTML rendering using Jinja2 to interact with users.
- Real-time prediction through API requests with image uploads.

### Example Performance Metrics:
| Metric             | Value | Details/Notes                     |
|--------------------|-------|------------------------------------|
| Training Samples   | 640   | Total training images             |
| Validation Samples | 160   | Total validation images           |
| Accuracy           | 0.90  | 90% correct predictions            |
| Precision (TB)     | 0.92  | 92% of TB predictions were correct|
| Recall (TB)        | 0.88  | 88% of actual TB cases detected   |

#### Class-wise Metrics:
- **Normal Image:** Precision = 0.49, Recall = 0.54, F1-score = 0.51
- **TB Image:** Precision = 0.49, Recall = 0.45, F1-score = 0.47

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage
### Training the Model
To train the model, run the following command:

```bash
python train.py
```

## Deploying the model using FastAPI:
To deploy the model, run the following command:

```bash
python main.py
```
The FastAPI application will be available at http://localhost:8000. You can upload chest X-ray images for classification.