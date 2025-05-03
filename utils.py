# utils.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data
def load_data(file_path="D:/Financial_fraud_detection_app/data/creditcard_2023.csv"):
    data = pd.read_csv(file_path)
    return data

# Function to preprocess data: clean, standardize, handle imbalance
def preprocess_data(data):
    # Drop missing values
    data = data.dropna()

    # Drop duplicate rows
    data = data.drop_duplicates()

    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    return X_resampled, y_resampled

# Function to split dataset
def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)

# Function for EDA: Plot Fraud vs Non-Fraud Pie Chart
def plot_fraud_distribution(data):
    labels = ['Non-Fraud', 'Fraud']
    counts = data['Class'].value_counts()
    colors = ['#4CAF50', '#F44336']
    plt.figure(figsize=(6,6))
    plt.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Fraud vs Non-Fraud Transactions')
    plt.axis('equal')
    plt.show()

# Function for EDA: Plot Correlation Heatmap
def plot_correlation_heatmap(data):
    plt.figure(figsize=(20,20))
    sns.heatmap(data.corr(), cmap='coolwarm', annot=False, linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()
