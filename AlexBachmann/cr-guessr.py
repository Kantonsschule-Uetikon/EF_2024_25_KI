import csv
import math

def daten_einlesen(dateiname):
    features = []
    labels = []
    with open(dateiname, newline='', encoding='utf-8') as datei:
        reader = csv.reader(datei)
        next(reader, None)
        for zeile in reader:
            features.append([zeile[0], float(zeile[5]), float(zeile[6])])
            labels.append(zeile[2])
    return features, labels

features, labels = daten_einlesen("AlexBachmann/dnd_monsters.csv")

right = 0
wrong = 0

for messwerte, label in zip(features, labels):
    x0 = 1
    x1 = messwerte[1]  # AC
    x2 = messwerte[2]  # HP
    w0 = -1.5
    w1 = 0.25
    w2 = 1/25
    
    actualcr = label

    z = x0 * w0 + x1 * w1 + x2 * w2
    z_new = math.floor(z)

    print(f"{messwerte[0]} with cr: {z_new} (actual cr: {actualcr})")
    
    if z_new == actualcr:
        print("right")
        right += 1
    elif z_new != actualcr:
        print("wrong")
        wrong += 1

    
accuracy = right/(right+wrong)
print(right)
print(wrong)