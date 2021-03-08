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
        """
        Test if the number of feature vectors is equal to number of labels after running split_feature_labels
        """
        self.mydata.read()
        self.mydata.split_features_labels()
        self.assertEqual(len(self.mydata.x), len(self.mydata.y))

    def test_normalize(self):
        """
        Test if the size of x remains the same after normalization
        """
        self.mydata.read()
        self.mydata.split_features_labels()
        tmp = self.mydata.x
        self.mydata.normalize()
        self.assertEqual(len(self.mydata.x), len(tmp))

    def test_split_train_val_test_1(self):
        """
        Test if the function produce a correct amount of x_train data based on test_ratio and val_ratio
        """
        self.mydata.x = [1,2,3,4,5,6,7,8,9,10]
        x_train = [1,2,3,4,5,6,7]
        self.mydata.split_train_val_test(test_ratio=20, val_ratio=10)
        self.assertEqual(self.mydata.x_train, x_train)

    def test_split_train_val_test_2(self):
        """
        Test if the function produce a correct amount of x_val data based on test_ratio and val_ratio
        """
        self.mydata.x = [1,2,3,4,5,6,7,8,9,10]
        x_val = [8]
        self.mydata.split_train_val_test(test_ratio=20, val_ratio=10)
        self.assertEqual(self.mydata.x_val, x_val)

    def test_split_train_val_test_3(self):
        """
        Test if the function produce a correct amount of x_test data based on test_ratio and val_ratio
        """
        self.mydata.x = [1,2,3,4,5,6,7,8,9,10]
        x_test = [9,10]
        self.mydata.split_train_val_test(test_ratio=20, val_ratio=10)
        self.assertEqual(self.mydata.x_test, x_test)

    def test_list_to_array(self):
        """
        Test if the type of data is numpy array after performing list_to_array() function
        :return: 
        """
        a = np.array([1])
        self.mydata.list_to_array()
        self.assertEqual(type(self.mydata.x_train), type(a))



if __name__ == '__main__':
    unittest.main()