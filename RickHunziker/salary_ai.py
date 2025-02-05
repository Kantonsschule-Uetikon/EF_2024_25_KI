# Test CSV-Datei einlesen
import csv

# Load the Salary dataset
data = []
with open("Salary_Dataset.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip header
    for row in reader:
        data.append(row)

print(data[:50])
