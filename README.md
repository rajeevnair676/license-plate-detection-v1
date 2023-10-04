# License plate detection and OCR - v1

## Problem Statement:
To detect the license plates and record the details from the detections using OCR. The goal is to provide the live details of the vehicles from the moving camera from the 360 patrol camera, with less latency

## Requirements:
The requirements are attached as a file in the repository with the name "requirements.txt". The python version is 3.11.0

## Approach
The approach was to train a machine learning model for the license plate detection, and then predict the bounding boxes for a new image, and then use the detected area to feed into an OCR engine, and retrieve the output. The development was done in phases:

### 1. Data Collection
The data for training the ML model for the license plate detection was collected from an external source, which was specifically for the UAE, with atleast 10k data points for training and 2.5k data points for validation. The details of the data is given below:

https://www.kaggle.com/datasets/rajeevnair676/yolo-license-roboflow

### 2. Building the model
The model was built with 2 approaches. Initially a model was trained to detect the cars in the images/videos, and train another model to segment the number plates from the car detection. The second approach was to train a license plate detection model directly on the images/videos. The best model of the 2 was chosen as the final model.

Model Architecture:
* A pretrained YOLOv8 model was used to train the license plate detection model as well as the car detection model

![image](https://github.com/rajeevnair676/license-plate-detection-v1/assets/97514601/40ba5872-b933-4429-8b85-ff4677cd94f8)

### 3. Training the model
The pretrained model was trained on the data from roboflow with around 10k data points onb training set and 2.5k on validation set. The training was done for 10 epochs, and the model weights were downloaded and saved. The model is saved in the folder 'Model\License_detection\best_licensedetect.pt'


