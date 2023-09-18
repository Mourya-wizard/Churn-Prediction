# Customer Churn Analysis

This Python script analyzes customer churn using a machine learning approach. The dataset used for this analysis is stored in a CSV file named 'customer_churn_large_dataset.csv'. The analysis includes data preprocessing, exploratory data analysis, model training, cross-validation, and hyperparameter tuning.

## Data Preprocessing

The first step is to load the dataset using the Pandas library and check for missing values. Missing values are dropped from the dataset. The data is also profiled using the `pandas-profiling` library, which generates a report in HTML format.

## Exploratory Data Analysis (EDA)

EDA involves visualizing the dataset to understand the relationships between variables. A correlation heatmap is created using `Seaborn` and `Matplotlib` to visualize the correlation between different features in the dataset.

## Label Encoding

Categorical variables in the dataset are one-hot encoded using Pandas' `get_dummies` function. This is done to convert categorical data into a numerical format suitable for machine learning models.

## Model Training

Two machine learning models, Random Forest Classifier and Logistic Regression, are trained on the dataset. The dataset is split into training and testing sets using `train_test_split`, and feature scaling is performed using `StandardScaler`. Model `accuracy`, `precision`, `recall`, and `F1-score` are evaluated.

## Cross Validation

Cross-validation is performed on the trained models using a list of classifiers, including `Logistic Regression`, `K-Nearest Neighbors`, `Random Forest`, and `XGBoost`. Cross-validation accuracies and accuracy scores are displayed for each model.

## Hyperparameter Tuning

Hyperparameter tuning is performed for each of the models using `GridSearchCV`. Different hyperparameters are defined for each model, and the best hyperparameters are selected to `optimize model performance`. The results of hyperparameter tuning are displayed, including the best hyperparameters and highest scores achieved for each model.

This script provides a comprehensive analysis of customer churn and demonstrates how to preprocess data, train machine learning models, and fine-tune model hyperparameters for better predictive performance.
