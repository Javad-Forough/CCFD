# CCFD project
## Introduction
This is the Credit Card Fraud Detection (CCFD) project made for the WASP course "Software Engineering and Cloud Computing".
In this project, First a real-world European Credit Card Dataset have been processed and normalized and become ready to feed into the machine learning model.
then, in the next step, the project implements a CCFD model based on Artificial Neural Network (ANN). It also evaluates the implemented model based on follwoing metrics:
- Precision
- Recall
- F-measure (F1)

And also it illustrates performance of the implemented ANN by plotting the Precision-Recall (PR) curve.
## How to run ccfd.py
To run the CCFD code, Please open a commandline shell in the same directory and enter the following line of command:
```
python ccfd.py
```
Or please open the ccfd.py file in an IDE and run it using that.
The expected final output of the computed metrics will be similar to the following:
```
Precision =  0.9803921568627451
Recall =  0.6666666666666666
F-measure =  0.7936507936507936
```
In addition, the resulted PR curve will be similar to the following:
![alt text](https://github.com/Javad-Forough/CCFD/blob/master/PR-curve.png)

## How to run test.py
For running the unittest implemented in test.py file, Please open a commandline shell in the same directory and enter the following line of command:
```
python -m unittest test.py
```
The expected final output of the unittest will be similar to the following:
```
Ran 12 tests in 39.908s
OK
```
