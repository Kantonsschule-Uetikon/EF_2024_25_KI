import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

# Load the dataset
data = pd.read_csv('pancreatic_cancer_prediction_sample.csv')

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Select the relevant features and the target variable
X = data[['Gender', 'Smoking_History', 'Alcohol_Consumption']].values
y = data['Survival_Status'].apply(lambda x: 1 if x == 'Survived' else 0).values  # Binary classification: 1 if Survived, else 0

# One-hot encode the categorical features
categorical_features = ['Gender', 'Smoking_History', 'Alcohol_Consumption']
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(data[categorical_features])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model
model = MLPClassifier(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', max_iter=1000, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (Survived)
accuracy = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, y_pred_proba)
print(f'Loss: {loss}, Accuracy: {accuracy*100:.2f}%')

# Display the predicted probabilities for the first few test samples
print(f'Predicted probabilities: {y_pred_proba[:10]}')