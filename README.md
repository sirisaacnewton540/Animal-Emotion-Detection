# Animal Emotion Detection

This project aims to detect animal emotions such as "Angry", "Sad", and "Happy" using a convolutional neural network (CNN) built with TensorFlow and Keras. The dataset comprises images categorized into three emotions, with separate directories for training, validation, and testing. <br> <br>
It also implements a graphical user interface (GUI) using Tkinter and OpenCV in Python for real-time animal emotion detection using a pre-trained deep learning model (animal_emotion_detection_model.keras). The application allows users to either open a camera for live detection or upload an image file for emotion analysis.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

Animal Emotion Detection System is designed to classify animal emotions into three categories: "Angry", "Sad", and "Happy". The model leverages a pre-trained VGG16 network and is fine-tuned to suit the specific requirements of this classification task. And further use the power of Tkinter and PIL for an interective GUI, where one can give commands whether he/she wants to detection the animal emotions using live feed or wanted to upload an image for detection. 

## Dataset

The dataset is compressed into zip file, so it is needed to be unzipped before use.
It is organized into three main directories:
- `train`: Contains 750 images (250 images per emotion category).
- `valid`: Contains 27 images (9 images per emotion category.
- `test`: Contains 30 images (10 images per emotion category.

Each image is of size 224x224 pixels in JPG format.

## Results
![Screenshot (631) (1)](https://github.com/user-attachments/assets/369dd949-675b-4d25-a6ed-af0ecc5955e8)
![Screenshot (633) (1)](https://github.com/user-attachments/assets/518f6ef1-d497-4210-bda6-f797d36c111e)
![Screenshot (632) (1)](https://github.com/user-attachments/assets/c83490b0-4e68-4e7d-adff-5d036595a023)

## Project Structure

animal-emotion-detection/ <br>
├── animal_emotions/ # Dataset directory <br>
├── models/ # Directory to save trained models <br>
├── scripts/ # Python scripts for training and evaluation <br>
├── gui/ # Python code for implementation of GUI <br>
├── README.md # Project documentation <br>
└── requirements.txt # Required Python packages <br>

## Installation

### Prerequisites

- Python 3.7 or higher
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Scikit-learn
- Pillow 8.3 or higher

### Install dependencies

To install the required packages, run: <br>
```
pip install -r requirements.txt
```

## Model Architecture
The model is based on the VGG16 architecture pre-trained on ImageNet. The top layers are replaced with custom layers suitable for the emotion detection task.

- Base Model: VGG16 (pre-trained on ImageNet)
- Custom Layers:
    1) Flatten
    2) Dense (512 units, ReLU activation)
    3) Dropout (50%)
    4) Dense (3 units, Softmax activation)

## Training
The model is trained using the following configuration:

- Optimizer: Adam
- Learning Rate: 0.0001
- Loss Function: Categorical Cross-Entropy
- Metrics: Accuracy
- Epochs: 50
- Batch Size: 32
- Early stopping and model checkpointing are used to prevent overfitting and save the best model.

## Evaluation
The model is evaluated on the test set. The test accuracy and a detailed classification report are generated.

## Results
The results section should summarize the test accuracy and key metrics such as precision, recall, and F1-score for each emotion category.

## Contributing
Contributions are welcome! Please fork this repository and submit pull requests for any enhancements or bug fixes.
