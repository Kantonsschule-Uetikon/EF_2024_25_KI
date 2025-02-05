import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
import pandas
import csv

def einlesen(dateiname):
    with open(dateiname) as datei:
        reader = csv.reader(datei)
        Daten = []
        for row in reader:
            Alter = row[0]
            Einkommen = row[1]
            HatGekauft = row[2]
            Daten.append((Alter, Einkommen, HatGekauft))
        return Daten

Daten = einlesen("Social_Network_Ads.csv")

X = Daten[0]
Y = Daten[1]
Z = Daten[2]
