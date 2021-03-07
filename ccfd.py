# Importing the necessary libraries
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, average_precision_score, recall_score, precision_score, accuracy_score, classification_report, precision_recall_curve,f1_score
from matplotlib import pyplot

# Reading the data
with open("Dataset/Data.txt", "rb") as fp:
    data_list = pickle.load(fp)

# Splitting the data into feature vectors and labels
x = []
y = []
for item in data_list:
    x.append(item[:-1])
    y.append(item[-1])

# Scaling the data and make it ready to be used by ML models
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
x = x.tolist()