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
ann_model = Sequential()

# Adding three layers to our ANN
ann_model.add(Dense(100, activation='relu'))
ann_model.add(Dense(10, activation='relu'))
ann_model.add(Dense(1, activation='sigmoid'))

#Compiling the ANN Model
ann_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model on the training data
ann_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1)

# predicting the class of each test instances
predictions = ann_model.predict_classes(x_test)
y_hat = predictions[:, 0]

# predicting the probability for each test instances
output_2 = ann_model.predict(x_test)[:, 0]

# Evaluating the model using Precision, Recall, and F1
precision = precision_score(y_test, y_hat)
recall = recall_score(y_test, y_hat)
f1 = f1_score(y_test, y_hat)

# Showing the evaluation result
print("Precision = ", precision)
print("Recall = ", recall)
print("F-measure = ", f1)

# showing Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, output_2)
pyplot.plot(recall, precision,color='blue', marker='.', label="ANN")
pyplot.show()