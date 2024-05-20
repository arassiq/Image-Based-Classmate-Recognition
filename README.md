# Image-Based-Classmate-Recognition

## Overview

This project focuses on detecting the presence of a classmate, Alex, in images using Convolutional Neural Networks (CNN). The model is trained on a custom dataset containing images of Alex and other individuals, allowing it to accurately identify Alex in new images. This project demonstrates the application of CNNs for real-world image detection tasks.

## Dataset

The dataset includes training and test images organized into two categories: `Alex` and `Not_Alex`. The training set is augmented to improve the model's robustness and accuracy.

## Methodology

### Data Preprocessing

The data preprocessing steps ensure that the images are ready for training and testing:

1. **Data Augmentation**: Applied to the training images to enhance model generalization.
   - Techniques: Rescaling, rotation, width and height shift, shear, zoom, and horizontal flip.

2. **Normalization**: Images are normalized to have pixel values in the range [0, 1].

### Model Architecture

A simple CNN model is built and trained using TensorFlow and Keras. The model architecture includes:

- Convolutional layers with ReLU activation and batch normalization.
- MaxPooling layers to reduce spatial dimensions.
- A fully connected (dense) layer for final classification.

### Training

The model is trained for 10 epochs using the training dataset. The training process includes compiling the model with the Adam optimizer and binary cross-entropy loss.

### Testing and Evaluation

The trained model is evaluated using a test image to determine whether Alex is present. The prediction is visualized by overlaying the result on the test image.

## Results

The project successfully detects the presence of Alex in images with high accuracy. The use of data augmentation and a robust CNN architecture contributes to the model's performance.

## Visualization

The test image is displayed with a title indicating whether Alex is detected. The title color changes based on the prediction result.

## Key Outcomes

Through this project, I demonstrate the ability to effectively detect a specific individual in images using CNNs. The methodology and results showcase the potential for applying similar techniques to various image recognition tasks.

## Acknowledgements

Special thanks to my professor for pushing me to challenge myself and the resources/libraries that made this project possible.
