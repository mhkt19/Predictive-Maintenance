
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

### Descriptive Statistics
| Feature               | Count  | Mean     | Std       | Min   | 25%   | 50%   | 75%   | Max   | Null Count |
|-----------------------|--------|----------|-----------|-------|-------|-------|-------|-------|------------|
| UDI                   | 10000  | 5000.5   | 2886.90   | 1     | 2500.75 | 5000.5 | 7500.25 | 10000 | 0          |
| Product ID            | 10000  | 4999.5   | 2886.90   | 0     | 2499.75 | 4999.5 | 7499.25 | 9999  | 0          |
| Type                  | 10000  | 1.1994   | 0.6002    | 0     | 1     | 1     | 2     | 2     | 0          |
| Air temperature [K]   | 10000  | 300.0049 | 2.0003    | 295.3 | 298.3 | 300.1 | 301.5 | 304.5 | 0          |
| Process temperature [K] | 10000  | 310.0056 | 1.4837    | 305.7 | 308.8 | 310.1 | 311.1 | 313.8 | 0          |
| Rotational speed [rpm] | 10000  | 1538.776 | 179.2841  | 1168  | 1423  | 1503  | 1612  | 2886  | 0          |
| Torque [Nm]           | 10000  | 39.9869  | 9.9689    | 3.8   | 33.2  | 40.1  | 46.8  | 76.6  | 0          |
| Tool wear [min]       | 10000  | 107.951  | 63.6541   | 0     | 53    | 108   | 162   | 253   | 0          |
| Target                | 10000  | 0.0339   | 0.1810    | 0     | 0     | 0     | 0     | 1     | 0          |

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
