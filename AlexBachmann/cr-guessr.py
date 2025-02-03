import csv
import math

def daten_einlesen(dateiname):
    features = []
    labels = []
    with open(dateiname, newline='', encoding='utf-8') as datei:
        reader = csv.reader(datei)
        next(reader, None)  # Skip header row if necessary
        for zeile in reader:  # Iterate over the reader
            features.append([zeile[0], float(zeile[5]), float(zeile[6])])
            labels.append(zeile[2])  # Ensure labels are floats
    return features, labels

features, labels = daten_einlesen("AlexBachmann/dnd_monsters.csv")

for messwerte, label in zip(features, labels):
    x0 = 0
    x1 = messwerte[1]  # AC
    x2 = messwerte[2]  # HP
    w0 = 1
    w1 = 0.25
    w2 = 1/25
    
    actualcr = label

    z = x0 * w0 + x1 * w1 + x2 * w2
    z = math.floor(z)
    
    print(f"{messwerte[0]} with cr: {z} (actual cr: {actualcr})")