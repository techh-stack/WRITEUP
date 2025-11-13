# ----------------------------------------------
# HEART DISEASE DATA OPERATIONS + CONFUSION MATRIX (with diagram)
# ----------------------------------------------

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Step 2: Load Dataset
data = pd.read_csv("C:\Heart.csv")
print("First 5 rows of data:\n", data.head(), "\n")

# Step 3: Perform Data Operations
print("Shape of Data:", data.shape, "\n")
print("Missing Values in each column:\n", data.isnull().sum(), "\n")
print("Data Types:\n", data.dtypes, "\n")
print("Count of Zeroes in each column:\n", (data == 0).sum(), "\n")
print("Mean Age of Patients:", data['Age'].mean(), "\n")

# Step 4: Extract Selected Columns
selected_data = data[['Age', 'Sex', 'ChestPain', 'RestBP', 'Chol']]
print("Selected Columns:\n", selected_data.head(), "\n")

# Step 5: Split Dataset (80% Training, 20% Testing)
train, test = train_test_split(selected_data, test_size=0.2, random_state=42)
print("Training Set Shape:", train.shape)
print("Testing Set Shape:", test.shape, "\n")

# Step 6: Confusion Matrix Example (COVID Test Case)
# Given Data:
# Predicted Positive = 100
# Actual Positive = 50
# True Positive (TP) = 45
# False Positive (FP) = 100 - 45 = 55
# False Negative (FN) = 50 - 45 = 5
# Total Samples = 500
# True Negative (TN) = 500 - (TP + FP + FN) = 395

# Create true and predicted values
y_true = [1]*50 + [0]*450        # 50 actual positives, 450 actual negatives
y_pred = [1]*45 + [0]*5 + [1]*55 + [0]*395  # predicted results

# Step 7: Confusion Matrix and Metrics
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm, "\n")

# Plot Confusion Matrix (Diagram)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix Diagram')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()

# Step 8: Calculate and Display Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Accuracy :", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall   :", round(recall, 3))
print("F1 Score :", round(f1, 3))

#new optional
tp, tn,fp,fn = 45,395,55,5
total = 500

acc= (tp+tn)/(total)
print(f'The  accuracy of model is:',acc)

prec= tp/(tp+fp)
print(f'The  precision of model is:',prec)

rec= tp/(tp+fn)
print(f'The  recall of model is:',rec)

f1_score = 2* (prec*rec)/(prec+rec)
print(f'The f1 score of model is:',f1_score)

# Step 9: Show only specific columns (e.g., Age and RestBP)
print("\nShowing only 'Age' and 'RestBP' columns:\n")
print(data[['Age', 'RestBP']].head())

print("\nShowing only 'Sex' and 'Chol' columns:\n")
print(data[['Sex','Chol']].head())
