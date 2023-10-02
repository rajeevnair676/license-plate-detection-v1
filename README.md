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





