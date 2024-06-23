
# Predictive Maintenance Classification

## Overview
This repository contains code for predicting machine failures and the type of failure using a synthetic dataset. The dataset reflects real predictive maintenance scenarios encountered in the industry.

## Dataset Description
The dataset consists of 10,000 data points stored as rows with 14 features in columns:

- **UID**: unique identifier ranging from 1 to 10000
- **Product ID**: consisting of a letter L, M, or H for low (50% of all products), medium (30%), and high (20%) as product quality variants and a variant-specific serial number
- **Air temperature [K]**: generated using a random walk process later normalized to a standard deviation of 2 K around 300 K
- **Process temperature [K]**: generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K
- **Rotational speed [rpm]**: calculated from power of 2860 W, overlaid with a normally distributed noise
- **Torque [Nm]**: torque values are normally distributed around 40 Nm with an σ = 10 Nm and no negative values
- **Tool wear [min]**: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process
- **Target**: Failure or Not
- **Failure Type**: Type of Failure

## Results
After running the model 10 times, the average results are as follows:

- **Failure Detection Accuracy**: 98.41%
- **Failure Type Detection Accuracy**: 98.155%

## Prerequisites

Ensure you have the following packages installed:

- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
## How to Run
1. Clone the repository.
2. Ensure you have the required packages installed:
3. Place the dataset in a folder named `dataset` in the root directory.
4. Run the script:
## Code Explanation
The code performs the following steps:
1. Loads and preprocesses the dataset.
2. Splits the data into training and testing sets.
3. Trains a Random Forest classifier on the training data.
4. Evaluates the classifier on the test data.
5. Repeats the training and evaluation process 10 times and calculates the average performance metrics.
6. Saves the results and confusion matrices in timestamped folders.

## Acknowledgements
- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification/data).
- Synthetic dataset reflects real predictive maintenance scenarios encountered in the industry.

## License
This project is licensed under the MIT License.
