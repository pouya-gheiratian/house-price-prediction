# House Price Prediction in Tehran

A Machine Learning project for predicting apartment prices in Tehran using the Linear Regression algorithm.

## Project Overview

This project focuses on predicting apartment prices in Tehran based on property characteristics such as area, number of rooms, location, and available facilities.

The project demonstrates a complete Machine Learning workflow, including data preprocessing, exploratory data analysis, feature engineering, model training, and model evaluation.

## Dataset Description

The dataset contains information about approximately 4,000 apartments in Tehran.

### Features

* Area (Square Meters)
* Number of Rooms
* Parking Availability
* Warehouse Availability
* Elevator Availability
* Address
* Price (Toman)

## Data Preprocessing

Several data quality issues were identified and handled before training the model:

* Removed records with missing address values
* Removed unrealistic area values (outliers)
* Cleaned inconsistent data entries
* Converted categorical features into numerical representations
* Prepared the dataset for machine learning algorithms

## Machine Learning Workflow

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Data Preparation
5. Train-Test Split
6. Linear Regression Model Training
7. Model Evaluation
8. House Price Prediction

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib

## Machine Learning Algorithm

* Linear Regression

## Dataset Notes

This dataset was originally collected approximately 3–4 years before the development of this project.

Therefore, the price values do not represent the current real estate market in Tehran. The dataset is used for educational and machine learning purposes, focusing on data preprocessing, feature engineering, model development, and prediction techniques.

## Objective

The goal of this project is to estimate apartment prices based on property characteristics and location-related features while applying fundamental Machine Learning concepts and best practices.

## Future Improvements

* Random Forest Regression
* XGBoost Regression
* Hyperparameter Tuning
* Advanced Feature Engineering
* Model Performance Optimization

## Author

Mohammad Pouya Gheiratian

