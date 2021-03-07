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

# Splitting the data into Train, Validation, and Test data
m = int(0.7 * len(x))
n = int(0.8 * len(x))
x_train = x[:m]
y_train = y[:m]
x_val = x[m:n]
y_val = y[m:n]
x_test = x[n:]
y_test = y[n:]

# Convert the data from List to numpy array
x_train=np.array(x_train)
y_train=np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)
x_test = np.array(x_test)

# Defining the ANN classifer model and adding the layers
clf = Sequential()