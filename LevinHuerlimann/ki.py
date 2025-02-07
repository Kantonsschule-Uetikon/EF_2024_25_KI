
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from sklearn import datasets
from sklearn.model_selection import train_test_split


df = pd.read_csv('LevinHuerlimann\Stroke_Prediction_Indians.csv') #csv lesen

df = df.drop(columns=['ID'])

gender_map = {
    "Female": 0,
    "Male": 1
}

work_type_map = {
    "Private": 1,
    "Self-employed": 2,
    "Government": 3,
    "Children": 4,
    "Never Worked": 5
}

residence_map = {
    "Rural": 0,
    "Urban": 1
}

smoking_map = {
    "Never smoked": 0,
    "Formerly smoked": 1,
    "Smokes": 2,
    "Unknown": 3
}

activity_map = {
    "Sedentary": 0,
    "Active": 1,
    "Light": 2,
    "Moderate": 3
}

diet_map = {
    "Non-Vegetarian": 0,
    "Vegetarian": 1,
    "Mixed": 2
}

education_map = {
    "Primary": 1,
    "Secondary": 2,
    "Tertiary": 3,
    "No education": 4
}

income_map = {
    "Low": 1,
    "Middle": 2,
    "High": 3
}

region_map = {
    "South": 0,
    "North": 1,
    "East": 2,
    "West": 3
}

df["Gender"] = df["Gender"].map(gender_map)
df["Work Type"] = df["Work Type"].map(work_type_map)
df["Residence Type"] = df["Residence Type"].map(residence_map)
df["Smoking Status"] = df["Smoking Status"].map(smoking_map)
df["Physical Activity"] = df["Physical Activity"].map(activity_map)
df["Dietary Habits"] = df["Dietary Habits"].map(diet_map)
df["Education Level"] = df["Education Level"].map(education_map)
df["Income Level"] = df["Income Level"].map(income_map)
df["Region"] = df["Region"].map(region_map)



X = df.drop('Stroke Occurrence', axis=1)
y = df['Stroke Occurrence']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
