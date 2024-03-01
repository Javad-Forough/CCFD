# Credit Card Fraud Detection Project

## Introduction
Welcome to the Credit Card Fraud Detection (CCFD) project developed for the WASP course "Software Engineering and Cloud Computing". This project focuses on detecting fraudulent transactions in credit card data using machine learning techniques.

In this project, a real-world European Credit Card Dataset has been processed and normalized to prepare it for input into the machine learning model. The subsequent step involves implementing a CCFD model based on Artificial Neural Network (ANN). The model's performance is evaluated using the following metrics:

- Precision
- Recall
- F-measure (F1)

Furthermore, the performance of the implemented ANN is visualized through the Precision-Recall (PR) curve.

## How to Run `ccfd.py`
To execute the CCFD code, follow these steps:

1. Open a command-line shell in the project directory.
2. Enter the following command:

```
python ccfd.py
```

Alternatively, you can open the `ccfd.py` file in an Integrated Development Environment (IDE) and run it from there.

The expected final output displaying the computed metrics will resemble the following:

```
Precision =  0.9803921568627451
Recall =  0.6666666666666666
F-measure =  0.7936507936507936
```

Additionally, the resulting PR curve will be similar to the following:

![Precision-Recall Curve](https://github.com/Javad-Forough/CCFD/blob/master/PR-curve.png)

## How to Run `test.py`
To execute the unit tests implemented in the `test.py` file, follow these steps:

1. Open a command-line shell in the project directory.
2. Enter the following command:

```
python -m unittest test.py
```

The expected final output of the unit test will resemble the following:

```
Ran 12 tests in 39.908s
OK
```

Feel free to explore and contribute to the project! If you encounter any issues or have suggestions for improvement, please don't hesitate to reach out.
