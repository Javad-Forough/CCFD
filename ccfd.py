# Importing the necessary libraries
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, average_precision_score, recall_score, precision_score, accuracy_score, classification_report, precision_recall_curve,f1_score
from matplotlib import pyplot


class Data:
    def __init__(self, path):
        self.path = path
        self.data = []
        self.x = []
        self.y = []
        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self.x_test = []
        self.y_test = []
        self.test_ratio = 80  # From 0 to 100
        self.val_ratio = 10  # From 0 to 100
        # Reading the data
        self.read(path)
        # Splitting the data into feature vectors and labels
        self.split_features_labels(self.data)
        # Scaling the data and make it ready to be used by ML models
        self.normalize()
        # Splitting the data into Train, Validation, and Test data
        self.split_train_val_test(self.test_ratio, self.val_ratio)
        # Convert the data from List to numpy array
        self.list_to_array()

    def read(self, path):
        with open(path, "rb") as fp:
            data_list = pickle.load(fp)
        self.data = data_list

    def split_features_labels(self, raw_data):
        for item in raw_data:
            self.x.append(item[:-1])
            self.y.append(item[-1])

    def normalize(self):
        scaler = StandardScaler()
        scaler.fit(self.x)
        self.x = scaler.transform(self.x)
        self.x = self.x.tolist()

    def split_train_val_test(self, test_ratio, val_ratio):
        m = int(((100 - (test_ratio + val_ratio))/100)*len(self.x))
        n = int(((100 - test_ratio)/100)*len(self.x))
        self.x_train = self.x[:m]
        self.y_train = self.y[:m]
        self.x_val = self.x[m:n]
        self.y_val = self.y[m:n]
        self.x_test = self.x[n:]
        self.y_test = self.y[n:]

    def list_to_array(self):
        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        self.x_val = np.array(self.x_val)
        self.y_val = np.array(self.y_val)
        self.x_test = np.array(self.x_test)


class Classifier:
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.predictions = []
        self.y_hat = []
        self.list_of_nodes = [100, 10, 1]
        self.list_of_activations = ['relu', 'relu', 'sigmoid']
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        # Defining the ANN classifer model and adding the layers
        self.build(self.list_of_nodes, self.list_of_activations)
        # Compiling the ANN Model
        self.compile()
        # Training the model on the training data
        self.train(self.x_train, self.x_val, self.y_train, self.y_val)
        # predicting the class of each test instances
        self.predict_class(self.x_test)
        # predicting the probability for each test instances
        self.predict_probs(self.x_test)
        # Evaluating the model using Precision, Recall, and F1
        self.evaluate(self.y_test, self.y_hat)
        # showing Precision-Recall curve
        self.show_pr_curve(self.y_test, self.ann_probs)

    def build(self, list_of_nodes, list_of_activations):
        self.list_of_nodes = list_of_nodes
        self.list_of_activations = list_of_activations
        self.ann_model = Sequential()
        # Adding layers to the ANN
        for i in range(len(list_of_nodes)):
            self.ann_model.add(Dense(list_of_nodes[i], activation=list_of_activations[i]))
        # self.ann_model.add(Dense(100, activation='relu'))
        # self.ann_model.add(Dense(10, activation='relu'))
        # self.ann_model.add(Dense(1, activation='sigmoid'))

    def compile(self):
        self.ann_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, x_train, x_val, y_train, y_val):
        self.ann_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1)

    def predict_class(self, x_test):
        self.predictions = self.ann_model.predict_classes(x_test)
        self.y_hat = self.predictions[:, 0]

    def predict_probs(self, x_test):
        self.ann_probs = self.ann_model.predict(x_test)[:, 0]

    def evaluate(self, y_test, y_hat):
        self.precision = precision_score(y_test, y_hat)
        self.recall = recall_score(y_test, y_hat)
        self.f1 = f1_score(y_test, y_hat)
        # Showing the evaluation result
        print("Precision = ", self.precision)
        print("Recall = ", self.recall)
        print("F-measure = ", self.f1)

    def show_pr_curve(self, y_test, ann_probs):
        precision, recall, _ = precision_recall_curve(y_test, ann_probs)
        pyplot.plot(recall, precision, color='blue', marker='.', label="ANN")
        pyplot.show()


if __name__ == "__main__":
    pass














