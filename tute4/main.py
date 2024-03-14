#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


def f(x: np.ndarray):
    return x**3 + 1


def question1():
    # Part a
    x = np.linspace(-1, 1, 100)
    # plt.rcParams['text.usetex'] = True

    plt.plot(x, f(x), label=r'x^3 + 1', color='b')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    # part b
    np.random.seed(50)
    # Generate random numbers
    training_size = 30
    train_x = np.random.uniform(-1, 1, size=training_size)
    noise = np.random.normal(size=training_size)
    train_y = f(train_x) + noise
    plt.plot(train_x, train_y, "+", color="red", label="Training Data")
    # Plot on top
    # plt.show()
    # Create Linear regressor
    # Recall that linear regression trains a series of weights applied to the
    # variables to minimise the loss function associated with (usually) the least squaes system
    reg = LinearRegression()
    # The data needs to be reshaped to factor in the DC term
    train_x = np.sort(train_x)
    train_x = train_x.reshape(-1, 1)
    print(train_x)
    reg.fit(train_x, train_y)
    # Evaluate the predictions for the training sets
    train_y_predict = reg.predict(train_x)
    training_error = mean_squared_error(train_y_predict, train_y)
    # Report the error
    print("Linear Regression MSE {}".format(training_error))
    # Plot the linear model
    plt.plot(train_x, train_y_predict,
             label="Linear Regression for {} Data Points".format(training_size))
    plt.legend()
    plt.grid(True)

    """
    PART D: Polynomial Regression
    Recall that polynomial regression adds another dimension to the input data points (The relevant
    x**n coordinate) and uses this to train the inforrmation
    """
    plt.figure(figsize=(12, 12))
    errors = []
    for degree in range(9):
        plt.subplot(3, 3, degree+1)
        # Create the polynomial regressor
        ply_feat = PolynomialFeatures(degree=degree)
        ply_x = ply_feat.fit_transform(train_x)
        # Use this information to predict the training results
        ply_reg = LinearRegression()
        # Fit the polynomial features to the linear regression model
        ply_reg.fit(ply_x, train_y)

        train_y_predict = ply_reg.predict(ply_x)
        # Compute the MSE for this
        errors.append(mean_squared_error(train_y_predict, train_y))

        # Plot this
        plt.plot(x, f(x))
        plt.plot(train_x, train_y, "+", label="Training Data")
        plt.plot(train_x, train_y_predict, label="Predicted Data")

        plt.grid(True)
        plt.title("Polynomial Regression (Degree = {})".format(degree))
    # reg.fit(train_x, train_y)

    plt.figure(figsize=(6, 6))
    plt.plot(np.arange(0, 9, 1), errors)
    plt.title('MSE')
    plt.xlabel('Degree')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    question1()
    exit(0)
