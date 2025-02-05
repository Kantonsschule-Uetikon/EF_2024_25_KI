import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('Tuberculosis_Dataset.csv')

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Assuming the last column is the target variable and the rest are features
X = data[['Country', 'Year', 'TB_Incidence_Rate', 'Population', 'HIV_Prevalence', 'Treatment_Success_Rate', 
          'Mortality_Rate', 'Urban_Population_Percentage', 'Smoking_Rate', 'Alcohol_Consumption', 
          'Healthcare_Expenditure_Per_Capita', 'Primary_Healthcare_Access', 'Age_Group', 'Gender', 
          'HIV_CoInfection_Rate', 'BCG_Vaccination_Coverage', 'Multidrug_Resistance_Rate', 'Unemployment_Rate', 
          'GDP_Per_Capita', 'Life_Expectancy', 'TB_Screening_Coverage', 'Rural_Population_Percentage']].values
y = data['TB_Incidence_Rate'].values

# One-hot encode the categorical features
categorical_features = ['Country', 'Year', 'Age_Group', 'Gender']
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(data[categorical_features])

# Combine the encoded categorical features with the rest of the features
X_rest = data.drop(columns=categorical_features + ['TB_Incidence_Rate']).values
X = np.hstack((X_encoded, X_rest))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model
model = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', max_iter=1000, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')