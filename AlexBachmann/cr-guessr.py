import csv
import tensorflow as tf
import pandas as pd

cr_trainer = pd.read_csv("dnd_monsters.csv", names=["name", "url", "cr", "type", "size","ac", "hp", "speed", "align", "legendary", "source", "str", "dex", "con", "int", "cha"])
cr_trainer.head()

cr_features = cr_trainer.copy()
cr_labels = cr_features.pop("cr")