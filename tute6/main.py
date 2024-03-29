#!/usr/bin/env python

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression


def question1():
    print("Question 1")
    # Load the data
    train_data = pd.read_csv(
        "ann-train.data", delim_whitespace=True, header=None)
    test_data = pd.read_csv(
        "ann-test.data", delim_whitespace=True, header=None)
    # The last value is the  y parameter
    y_train = train_data.iloc[:, -1]
    y_test = test_data.iloc[:, -1]
    X_train = train_data.drop(train_data.columns[-1], axis=1)
    X_test = test_data.drop(test_data.columns[-1], axis=1)
    # Train the Learning Model
    k = 5
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    # Create predictions
    y_train_predict = classifier.predict(X_train)
    y_test_predict = classifier.predict(X_test)

    # Rank the model
    train_score = accuracy_score(y_train_predict, y_train)
    test_score = accuracy_score(y_test_predict, y_test)
    print("Train Score {}".format(train_score))
    print("Test Score {}".format(test_score))
    # Generate the confusion_matrix, This describes the perf of the model with respect
    # To the True Positives False Negatives, False Positives and True Negatives
    conf_matrix = confusion_matrix(y_test_predict, y_test)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix, display_labels=classifier.classes_)
    disp.plot(cmap="Reds")
    plt.title("Thyroid KNN-{} Confusion Matrix".format(k))
    plt.show()


def question2():
    train_data = pd.read_csv(
        "ann-train.data", delim_whitespace=True, header=None)
    test_data = pd.read_csv(
        "ann-test.data", delim_whitespace=True, header=None)
    # Check the sizes of these
    print("Training Data Siz {}".format(train_data.shape))
    print("Test Data Siz {}".format(test_data.shape))
    # The problem of false positives can cause issues. Modify the model such that the irregular data sets are scored more harshly
    # A value of three or two is considered abnormal. (Hypo / Hyper thyroid) The value of 0 is considered normal

    # Convert this data set to that
    for index, row in train_data.iterrows():
        # Check if the row value is equal to three change it to two if so
        if (row.iloc[-1] == 3):
            train_data.iloc[index, -1] = 2

    for index, row in test_data.iterrows():
        # Check if the row value is equal to three change it to two if so
        if (row.iloc[-1] == 3):
            test_data.iloc[index, -1] = 2
    # Create Logistic Regressor
    y_train = train_data.iloc[:, -1]
    X_train = train_data.drop(train_data.columns[-1], axis=1)
    y_test = test_data.iloc[:, -1]
    X_test = test_data.drop(test_data.columns[-1], axis=1)
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    # Predict the test data
    y_test_predict = log_reg.predict(X_test)

    # evaluate and display the Confusion matrix
    conf_mat = confusion_matrix(y_test_predict, y_test)
    accuracy = accuracy_score(y_test_predict, y_test)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=["Normal", "Abnormal"])
    disp.plot(cmap="Reds")
    plt.title("Thyroid Confusion Matrix using Logistic Regressor")
    print("Accuracy {}".format(accuracy))
    plt.show()
    # Use the probability functions
    threshold = 0.05
    prob_test = log_reg.predict_proba(X_test)
    print(prob_test)
    print(y_test_predict)
    # Loop over the proabilities, if the value is greater than threshold assign to
    y_prob_thresh = np.array([])
    for index, row in X_test.iterrows():
        # Inspect the probability
        prob_1 = prob_test[index][0]
        if (prob_1 > threshold):
            y_prob_thresh = np.block([y_prob_thresh, 1])
        else:
            y_prob_thresh = np.block([y_prob_thresh, 2])
    print(y_prob_thresh)

    # Recalculate confusion matrix
    new_conf_mat = confusion_matrix(y_prob_thresh, y_test)
    # Display the new confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=new_conf_mat, display_labels=["Normal", "Abnormal"])
    disp.plot(cmap="Reds")
    plt.title(
        "Thyroid Confusion Matrix with Logistic Regression (r = {})".format(threshold))
    print("Accuracy with r = {}: {}".format(
        threshold, accuracy_score(y_prob_thresh, y_test)))
    plt.show()


if __name__ == "__main__":
    # question1()
    question2()
