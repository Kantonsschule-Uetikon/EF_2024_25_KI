import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the Salary dataset
data = [] # used as a matrix
with open("Salary_Dataset.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip header
    for row in reader:
        data.append(row)

data = np.array(data) # Easier to work with np arrays

print(data[:5])

# Convert gender ('Male' and 'Female' into numbers)
gender = data[:, 1] # The second column of every row
encoder = LabelEncoder()
gender_encoded = encoder.fit_transform(gender)
data[:, 1] = gender_encoded

print(data[:5])
