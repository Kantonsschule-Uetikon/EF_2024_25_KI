import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout

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

# One-hot encode the target variable
# One-hot encoding -> convert each category into a binary vector
encoder_one_hot = OneHotEncoder(sparse_output=False)
y_one_hot = encoder_one_hot.fit_transform(y.reshape(-1, 1)) 

# Split data into training and testing sets
# 80% of data for training; 20% for testing
# random_state to ensure categorisation always the same  (reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Input(shape=(5,))) # 5 features
model.add(Dense(32, activation='relu')) # relu = rectified linear unit; negative values = 0
model.add(Dropout(0.2)) # get rid of 20 % to prevent overfitting
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='softmax')) # 4 salary categories

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
