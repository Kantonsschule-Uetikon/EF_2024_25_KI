import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

# Load the dataset
data = pd.read_csv('updated_pancreatic_cancer_data.csv')

# Replace 0 with 'no' and 1 with 'yes'
data = data.replace({0: 'no', 1: 'yes'})

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Select the relevant features and the target variable
X = data[['Gender', 'Smoking_History', 'Alcohol_Consumption', 'Age']]
y = data['Survival_Status'].apply(lambda x: 1 if x == 'yes' else 0)

# Map Gender to numerical values
X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})

# One-hot encode the categorical features
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X[['Smoking_History', 'Alcohol_Consumption']])

# Combine the encoded categorical features with the rest of the features
X_rest = X[['Gender', 'Age']].values
X = np.hstack((X_encoded, X_rest))

# Split the dataset into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build and train the neural network model
model = MLPClassifier(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (Survived)
accuracy = accuracy_score(y_test, model.predict(X_test))
loss = log_loss(y_test, y_pred_proba)
print(f'Loss: {loss}, Accuracy: {accuracy*100:.2f}%')

# Display the predicted probabilities for the first few test samples
print("\nDetailed information for the first 10 test samples:")
for i in range(10):
    sample = X_test[i]
    gender = 'Male' if sample[-2] == 1 else 'Female'
    age = sample[-1]
    smoking_history = 'yes' if sample[0] == 1 else 'no'
    alcohol_consumption = 'yes' if sample[1] == 1 else 'no'
    probability = y_pred_proba[i] * 100
    print(f'Sample {i+1}: Gender: {gender}, Age: {age}, Smoking History: {smoking_history}, Alcohol Consumption: {alcohol_consumption}, Probability of surviving pancreatic cancer: {probability:.2f}%')