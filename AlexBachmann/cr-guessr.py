import csv
import math

def daten_einlesen(dateiname):
    features, labels = [], []
    with open(dateiname, newline='', encoding='utf-8') as datei:
        reader = csv.reader(datei)
        next(reader, None)
        for zeile in reader:
            features.append([zeile[0], float(zeile[5]), float(zeile[6])])
            labels.append(zeile[2])
    return features, labels

features, labels = daten_einlesen("AlexBachmann/dnd_monsters.csv")

cr_mapping = {
    "0": -3, "1/8": -2, "1/4": -1, "1/2": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18, "19": 19, "20": 20, "21": 21, "22": 22, "23": 23, "24": 24, "25": 25, "26": 26, "27": 27, "28": 28, "29": 29, "30": 30
}

max_ac = max(f[1] for f in features)
max_hp = max(f[2] for f in features)

w0 = 0.1
w1 = 0.1
w2 = 0.1
right = 0
wrong = 0

for messwerte, label in zip(features, labels):
    if label not in cr_mapping:
        continue

    y = cr_mapping[label]
    x0, x1, x2 = 1, messwerte[1] / max_ac, messwerte[2] / max_hp
    z = x0 * w0 + x1 * w1 + x2 * w2
    z_new = max(min(math.floor(z), 30), 0)

    actualcr = label
    cr = {"1/8": 0.125, "1/4": 0.25, "1/2": 0.5}.get(actualcr, float(actualcr))

    learning_rate, max_iterations, iterations = 0.005, 10000, 0

    while y != z_new and iterations < max_iterations:
        prev_w0, prev_w1, prev_w2 = w0, w1, w2
        z = x0 * w0 + x1 * w1 + x2 * w2

        w0 += (y - z) * x0 * learning_rate
        w1 += (y - z) * x1 * learning_rate
        w2 += (y - z) * x2 * learning_rate

        z = x0 * w0 + x1 * w1 + x2 * w2
        z_new = max(min(math.floor(z), 30), 0)

        iterations += 1

    print(f"{messwerte[0]} with predicted CR: {z_new} (actual CR: {actualcr})")

    if cr == z_new:
        right += 1
    elif cr == z_new + 1:
        right += 1
    elif cr == z_new - 1:
        right += 1
    else:
        wrong += 1

print(f"Correct: {right}, Incorrect: {wrong}")
accuracy = right/(right+wrong) * 100
print(f"Accuracy: {accuracy}")
