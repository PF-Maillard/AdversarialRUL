# Adversarial Predictive Maintenance

## About

This repository contains the implementation code for the "Adversarial Predictive Maintenance". The repository is dedicated to developing adversarial methods to deceive predictive maintenance systems. The implemented methods include the Fast Gradient Sign Method (FGSM), Basic Iterative Method (BIM), Carlini & Wagner (CW) attack L2, and the Minimal-L0 method (L0). The tests have been done considering the dataset: NASA Turbofan Jet Engine Data Set.

Note the utilization of the following repository: 
- RUL Predictions using PyTorch LSTM: https://www.kaggle.com/code/jinsolkwon/rul-predictions-using-pytorch-lstm/notebook 

## Procedure

You must manually download the NASA Turbofan Jet Engine dataset and place it in the "DatasetRUL" folder. The dataset is available here: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps. We only consider the first TurboFan dataset for experiments

Included is one primary notebook:
- Main.ipynb

This notebook is designed to automatically fetch the necessary libraries, allowing for use without any additional prerequisites.

## License

This project is licensed under the MIT License. The full text of the MIT License can be found in the LICENSE.md file at the root of this project.

