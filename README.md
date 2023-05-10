<h1>Smart Home AI</h1>

This repository contains code and data for training an AI model to control smart home devices based on environmental factors.

# Overview

The goal of this project is to build an AI model that can learn from the historical data of your smart home and make decisions on how to control your home devices. It includes multiple light switches, RGB LEDs, and outlets.

# The AI model uses data such as:

Indoor temperature
Outdoor temperature
Light level
Motion detection status
Status of multiple light switches
RGB values of an LED light
Status of multiple outlets
Dataset

The dataset is a CSV file that records the statuses of the devices and the environmental factors in your smart home. Each row in the CSV file represents a snapshot of your smart home at a certain timestamp.

An example of the dataset can be found in the smart_home_data.csv file in this repository.

# Requirements

Python 3.6 or later
Tensorflow 2.0 or later
Scikit-learn
Pandas

# How to Use

Clone this repository to your local machine.
Install the required Python packages.
Run the train_model.py script to train the AI model.
The trained model will be saved to disk and can be used to make predictions.
License

This project is licensed under the terms of the MIT license.