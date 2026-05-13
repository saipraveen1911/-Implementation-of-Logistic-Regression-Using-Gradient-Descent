# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook


## Algorithm

1.Load dataset and convert placement status to numeric.

2.Select numeric features and split into train/test sets.

3.Initialize weights and bias to zero.

4.Apply sigmoid function and update weights using gradient descent.

5.Predict test data and compute accuracy. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SAI PRAVEEN C
RegisterNumber:  212225230239
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the data 
# Make sure the file name matches exactly what is in your folder
try:
    data = pd.read_csv("Placement_Data (1).csv")
    print("File loaded successfully!")
except FileNotFoundError:
    print("Error: The file 'Placement_Data (1).csv' was not found in this folder.")
    print("Please check Method 1 or 2 in the previous message to upload it.")

# 2. Preprocessing
data["status"] = data["status"].map({"Placed": 1, "Not Placed": 0})

# Select independent variables and target variable
X = data[["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]].values
y = data["status"].values

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Feature Scaling (Important for Gradient Descent)
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# 5. Logistic Regression from Scratch
n_features = X_train.shape[1]
w = np.zeros(n_features)
b = 0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Hyperparameters
lr = 0.01
epochs = 1000

# Training (Gradient Descent)
for i in range(epochs):
    z = np.dot(X_train, w) + b
    y_pred = sigmoid(z)

    dw = (1/len(X_train)) * np.dot(X_train.T, (y_pred - y_train))
    db = (1/len(X_train)) * np.sum(y_pred - y_train)

    w -= lr * dw
    b -= lr * db

# 6. Prediction Function
def predict(X):
    z = np.dot(X, w) + b
    probs = sigmoid(z)
    return (probs >= 0.5).astype(int)

# 7. Evaluate the Model
y_test_pred = predict(X_test)

accuracy = np.mean(y_test_pred == y_test)
print("\nAccuracy:", round(accuracy, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
```

## Output:

<img width="580" height="360" alt="ML EXP - 6" src="https://github.com/user-attachments/assets/c228d11d-f5a4-4983-a37f-c85b76b3ad83" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

