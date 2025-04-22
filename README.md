# Drowsiness Detection using CNN

## Overview
Drowsiness detection is a crucial application in driver safety systems to prevent accidents caused by fatigue. This project implements a Convolutional Neural Network (CNN) to classify images of a driver's face into different states:
- **Open Eyes**
- **Closed Eyes**
- **Yawning**
- **Not Yawning**

By analyzing these features, the model determines whether the driver is drowsy and raises an alert if necessary.

## Features
- Real-time detection of drowsiness.
- CNN-based classification model.
- Trained on a dataset containing images of open/closed eyes and yawning/not yawning.
- Can be integrated with real-time video streams.
- Provides an alert mechanism when drowsiness is detected.

## Dataset
The dataset consists of labeled images capturing different facial states:
- **Open Eyes**: Normal alert state.
- **Closed Eyes**: Indicates possible drowsiness.
- **Yawning**: Signs of fatigue.
- **Not Yawning**: Normal state.

To run the project download the dataset from the below link

https://drive.google.com/drive/folders/1Sn2ZZ4GXwTHQLvC8cvM2tgwf-fOeItUL?usp=sharing


## Model Architecture
The model is built using a deep CNN with multiple convolutional layers followed by dense layers for classification. The key components are:
- **Convolutional Layers**: Extract spatial features from images.
- **Pooling Layers**: Reduce dimensionality and improve efficiency.
- **Fully Connected Layers**: Perform final classification.
- **Softmax Activation**: Outputs probability distribution for class labels.

## Requirements
To run this project, install the following dependencies:
```bash
pip install tensorflow keras numpy opencv-python matplotlib
```

## Training the Model
Run the following script to train the CNN model:
```bash
python model.py
```
The trained model will be saved for later inference.

## Running the Drowsiness Detection System
To test the system on a real-time webcam feed, execute:
```bash
python drowsiness_detection.py
```
This will open a webcam window, process frames, and detect drowsiness in real-time.

## Results
The model achieves high accuracy of 96% in detecting drowsiness and provides reliable predictions for different facial states.

## Future Improvements
- Enhance dataset size and diversity.
- Improve real-time processing speed.
- Implement a mobile application for on-the-go monitoring.

## Conclusion
This project serves as an effective safety mechanism for preventing accidents caused by drowsy driving. By leveraging deep learning, it provides accurate real-time detection and can be further improved with advanced techniques.

