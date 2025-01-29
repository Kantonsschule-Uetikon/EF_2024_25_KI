import numpy as np

daten = np.genfromtxt("marriage_divorce_india_with_id.csv", delimiter=",", dtype=str, skip_header=1)

saubere_daten = daten[:, [1, 2, 3, 5, 7, 8, 9, 10]]

print(saubere_daten)