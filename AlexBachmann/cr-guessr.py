import csv

def daten_einlesen(dateiname):
    features = []
    labels = []
    with open(dateiname) as datei:
        for zeile in csv.reader(datei):  # Jede Zeile
            features.append([zeile[0],zeile[5],zeile[6]])
            labels.append(zeile[2])
    return features, labels

features, labels = daten_einlesen("AlexBachmann/dnd_monsters.csv")

for messwerte, label in zip (features, labels):
    #print(f"Eine {label} mit den Messwerten {messwerte}.")
    
    x0 = 0
    x1 = float(messwerte[1]) #ac
    x2 = float(messwerte[2]) #hp
    w0 = 1
    w1 = 1
    w2 = 1
    
    z = x0*w0 + x1*w1 + x2*w2
    
    print(z)