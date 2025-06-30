# Mobile-Price-Range-Prediction-Using-Machine-Learning
This project aims to predict the price range of mobile phones based on their technical specifications using supervised machine learning classification algorithms.

Problem Statement
Given various features of mobile devices such as battery power, RAM, display resolution, internal memory, and more, the goal is to classify the phone into one of four price categories:

0: Low cost

1: Medium cost

2: High cost

3: Very high cost

Objectives
Perform data cleaning and preprocessing

Conduct Exploratory Data Analysis (EDA)

Visualize feature distributions and relationships

Engineer new relevant features

Train classification models including Logistic Regression, K-Nearest Neighbors, and Random Forest

Tune hyperparameters for optimal model performance

Evaluate and compare model performance metrics

Dataset
The dataset includes 20+ features such as:

battery_power, ram, px_height, px_width

int_memory, mobile_wt, n_cores, sc_h, sc_w, etc.
The target variable is price_range, which represents the category of the phone.

Tools and Libraries
Python, Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn (LogisticRegression, RandomForest, KNN, GridSearchCV)

Models Used
Logistic Regression

K-Nearest Neighbors (KNN)

Random Forest Classifier

Hyperparameter tuning with GridSearchCV
