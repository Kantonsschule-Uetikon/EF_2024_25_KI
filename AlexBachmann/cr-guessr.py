import csv
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np

cr_trainer = pd.read_csv("dnd_monsters.csv", names=["name", "url", "cr", "type", "size","ac", "hp", "speed", "align", "legendary", "source", "str", "dex", "con", "int", "cha"])
cr_trainer.head()

cr_features = cr_trainer.copy()
cr_labels = cr_features.pop("cr")
cr_features = np.array(cr_features)
cr_features

cr_model = tf.keras.Sequential([
  layers.Dense(64, activation='relu'),
  layers.Dense(1)
])

cr_model.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam())

cr_model.fit(cr_features, cr_labels, epochs=10)

normalize = layers.Normalization()

norm_cr_model = tf.keras.Sequential([
  normalize,
  layers.Dense(64, activation='relu'),
  layers.Dense(1)
])

norm_abalone_model.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam())

norm_abalone_model.fit(cr_features, cr_labels, epochs=10)