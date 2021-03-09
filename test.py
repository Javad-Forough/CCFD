import unittest
from ccfd import Data, Classifier
import numpy as np

class TestData(unittest.TestCase):
    def setUp(self):
        self.mydata = Data("Dataset/Data.txt")

    def test_read(self):
        """
        Test if after running the method read, type of mydata.data is List
        """
        self.mydata.read()
        self.assertEqual(type(self.mydata.data), list)

    def test_split_features_labels(self):

        self.mydata.read()
        self.mydata.split_features_labels()
        self.assertEqual(len(self.mydata.x), len(self.mydata.y))

    def test_normalize(self):

        self.mydata.read()
        self.mydata.split_features_labels()
        tmp = self.mydata.x
        self.mydata.normalize()
        self.assertEqual(len(self.mydata.x), len(tmp))

    def test_split_train_val_test_1(self):

        self.mydata.x = [1,2,3,4,5,6,7,8,9,10]
        x_train = [1,2,3,4,5,6,7]
        self.mydata.split_train_val_test(test_ratio=20, val_ratio=10)
        self.assertEqual(self.mydata.x_train, x_train)

    def test_split_train_val_test_2(self):

        self.mydata.x = [1,2,3,4,5,6,7,8,9,10]
        x_val = [8]
        self.mydata.split_train_val_test(test_ratio=20, val_ratio=10)
        self.assertEqual(self.mydata.x_val, x_val)

    def test_split_train_val_test_3(self):

        self.mydata.x = [1,2,3,4,5,6,7,8,9,10]
        x_test = [9,10]
        self.mydata.split_train_val_test(test_ratio=20, val_ratio=10)
        self.assertEqual(self.mydata.x_test, x_test)

    def test_list_to_array(self):

        a = np.array([1])
        self.mydata.list_to_array()
        self.assertEqual(type(self.mydata.x_train), type(a))

    def test_isinstance(self):

        self.assertIsInstance(self.mydata, Data)


class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.mydata = Data("Dataset/Data.txt")
        self.mydata.read()
        self.mydata.split_features_labels()
        self.mydata.normalize()
        self.mydata.split_train_val_test(test_ratio=20, val_ratio=10)
        self.mydata.list_to_array()
        self.annmodel = Classifier(self.mydata.x_train, self.mydata.y_train,
                                   self.mydata.x_val, self.mydata.y_val, self.mydata.x_test,
                                   self.mydata.y_test)

    def test_build(self):

        list_of_nodes = [100, 10, 1]
        list_of_activations = ['relu', 'relu', 'sigmoid']
        self.annmodel.build(list_of_nodes, list_of_activations)
        self.assertEqual(len(self.annmodel.ann_model.layers), len(list_of_nodes))

    def test_predict_class(self):

        self.annmodel.build(list_of_nodes=[100, 10, 1], list_of_activations=['relu', 'relu', 'sigmoid'])
        self.annmodel.compile()
        self.annmodel.train(epochs=1)
        self.annmodel.predict_class()
        self.assertEqual(len(self.annmodel.y_hat), len(self.annmodel.x_test))

    def test_predict_probs(self):

        self.annmodel.build(list_of_nodes=[100, 10, 1], list_of_activations=['relu', 'relu', 'sigmoid'])
        self.annmodel.compile()
        self.annmodel.train(epochs=1)
        self.annmodel.predict_probs()
        self.assertEqual(len(self.annmodel.ann_probs), len(self.annmodel.x_test))

    def test_isinstance(self):

        self.assertIsInstance(self.annmodel, Classifier)


if __name__ == '__main__':
    unittest.main()