import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
import pandas
import csv

# Open the file in read mode
#Daten = pandas.read_excel('Social_Network_Ads.xls', engine='xlrd')
#Alter = pandas.read_excel('Social_Network_Ads.xls', usecols=['A'], engine='xlrd')
#Einkommen = pandas.read_excel('Social_Network_Ads.xls', usecols=['B'], engine='xlrd')
#Hat_Gekauft = pandas.read_excel('Social_Network_Ads.xls', usecols=['C'], engine='xlrd')

def einlesen(dateiname):
    with open(dateiname) as datei:
        return [(Age,EstimatedSalary,Purchased) for (Age,EstimatedSalary,Purchased) in csv.reader(datei)]

einlesen("Social_Network_Ads.csv")

print(Age)

