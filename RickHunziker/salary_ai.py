import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Load the Salary dataset
data = [] # used as a matrix
with open("Salary_Dataset.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip header
    for row in reader:
        data.append(row)

data = np.array(data) # Easier to work with np arrays

# Convert gender ('Male' and 'Female' into '1' and '0')
gender = data[:, 1] # The second column of every row
encoder_Label = LabelEncoder()
gender_encoded = encoder_Label.fit_transform(gender)
data[:, 1] = gender_encoded

# Convert Age, Gender, Education Level, Years of Experience, Senior to float;
# Create matrix with features x datasets
X = data[:, [0, 1, 2, 4, 8]].astype(float)

# Convert Salary to float;
y = data[:, 5].astype(float)

print(y)

# Group Salary into 4 categories -> Labels
for i in range(len(y)):
    if y[i] < 70000:
        y[i] = 0
    elif y[i] < 115000:
        y[i] = 1
    elif y[i] < 160000:
        y[i] = 2
    else:
       y[i] = 3

print(y)

# One-hot encode the target variable
# One-hot encoding -> convert each category into a binary vector
encoder_one_hot = OneHotEncoder(sparse_output=False)
y_one_hot = encoder_one_hot.fit_transform(y.reshape(-1, 1))

print(y_one_hot)
