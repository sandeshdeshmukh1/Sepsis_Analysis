# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import os
try:
    # %tensorflow_version only exists in Colab.
    #   %tensorflow_version 2.x
    print('2x')
except Exception:
    pass
import tensorflow as tf
# tf.random.set_seed(1)
import seaborn as sns
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras import regularizers

model = tf.keras.models.load_model('Final_Model/sepsis_improved.h5')
model.summary()

# pre = pd.read_csv('Final_Model/p000053.psv',delimiter='|')
pre = pd.read_csv('Final_Model/p100004.psv', delimiter='|')  # no sepesis
length = pre.shape[0]
print(length)

pre = pre.interpolate()
pre = pre.fillna(method='bfill')
pre = pre.fillna(method='ffill')
pre = pre.fillna(pre.mean())
# pre.isna().sum()


# X_HR = np.array(pre['HR'])
# X_Temp = np.array(pre['Temp'])
train = pre
X_HR = np.array(pre['HR'])  # (no_sampels,60,no_features)
X_Temp = np.array(pre['Temp'])  # (no_sampels,60,no_features)
X_Resp = np.array(pre['Resp'])
X_WBC = np.array(pre['WBC'])
X_MAP = np.array(pre['MAP'])
X_Gender = np.array(pre['Gender'])
X_Age = np.array(pre['Age'])
# y = train.groupby('Patient_id')['SepsisLabel_2'].apply(list)
y_original = pre['SepsisLabel']

temp = np.column_stack((X_HR, X_Temp, X_Resp, X_WBC, X_MAP, X_Gender, X_Age))
l = len(temp)

f = ['X_HR', 'X_Temp', 'X_Resp', 'X_WBC', 'X_MAP', 'X_Gender', 'X_Age']

padded_ip = tf.keras.preprocessing.sequence.pad_sequences(
    [temp], maxlen=60, padding='post', value=-1)
y_pad = tf.keras.preprocessing.sequence.pad_sequences(
    [y_original], maxlen=60, padding='post', value=-1)
# print(padded_ip)
ip = padded_ip.reshape(-1, 60, len(f))

ip.shape

y1 = model.predict(padded_ip)
for i, j in zip(y1[0], y_original):
    print(i, j)

y1 = model.predict(padded_ip)
for i, j in zip(y1[0], y_original):
    print(i, j)
