import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data                        # Matrix mit Features × Datensätze
y = iris.target.reshape(-1, 1)       # Labels

# One-hot encode the target variable
encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Define the model with increased max_iter and decreased tol
model = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', solver='adam', max_iter=1000, tol=1e-4, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, model.predict_proba(X_test))
print(f'Loss: {loss}, Accuracy: {accuracy}')
