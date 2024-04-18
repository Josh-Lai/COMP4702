#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def create_train_test_data(data_size=10, test_size=0.3):
    data = pd.read_csv('w3classif.csv', names=["X1", "X2", "Y"])
    X = data.drop("Y", axis=1)
    y = data["Y"]
    seed = np.random.RandomState(80)

    X_trains, X_tests = [], []
    y_trains, y_tests = [], []
    print("Generating test data with Test size = {}".format(test_size))
    for i in range(0, data_size):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed)
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
    return X_trains, X_tests, y_trains, y_tests


def question2():
    # Read the training and test sets, train a KNN classifier for them
    split = 0.9
    test_num = 10
    x_trains, x_tests, y_trains, y_tests = create_train_test_data(
        test_num, split)
    numTests = len(x_trains)
    print("Read {} tests".format(numTests))
    # Loop over the data sets and train classifier
    k = 3
    train_acc, test_acc = [], []
    train_acc_avg = 0
    test_acc_avg = 0
    for i in range(len(x_trains)):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_trains[i], y_trains[i])
        # Predict for x_tests and training data
        y_test_predict = knn.predict(x_tests[i])
        y_train_predict = knn.predict(x_trains[i])

        # Find the misclassification rates
        train_misclass = (1 - knn.score(x_trains[i], y_trains[i])) * 100
        test_misclass = (1 - knn.score(x_tests[i], y_tests[i])) * 100
        train_acc_avg += train_misclass
        test_acc_avg += test_misclass
        train_acc.append(train_misclass)
        test_acc.append(test_misclass)
    test_acc_avg /= len(x_trains)
    train_acc_avg /= len(x_trains)
    # Compute the standard deviation from this data set
    test_acc_std_dev = np.std(np.array(train_acc))
    train_acc_std_dev = np.std(np.array(test_acc))
    plt.figure()
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.legend(["Training Accuracy", "Test Accruacy"])
    plt.xlabel("Test No.")
    plt.ylabel("Misclassification Rate")
    plt.title("Accuracy for {} trials with {}% training data".format(
        test_num, split * 100))
    plt.grid(True)
    plt.show()

    print("Test misclassification Average {}%".format(test_acc_avg))
    print("Training misclassification Average {}%".format(train_acc_avg))
    print("Test Misclassification std dev {}".format(test_acc_std_dev))
    print("Train Misclassification std dev {}".format(train_acc_std_dev))


def question4(num_neighbours=3, num_folds=10):
    # Perform cross validation on the data set for data set based on the number of neighbours
    seed = np.random.RandomState(80)
    data = pd.read_csv('w3classif.csv', names=["X1", "X2", "Y"])
    # Shuffle the data
    data = shuffle(data, random_state=seed)
    X = data.drop("Y", axis=1)
    y = data["Y"]

    knn = KNeighborsClassifier(n_neighbors=num_neighbours)
    scores = cross_val_score(knn, X, y, cv=num_folds)
    score_mean = np.mean(scores)
    score_std_dev = np.std(scores)
    print("Score Mean: {}".format(score_mean))
    print("Score Std Dev: {}".format(score_std_dev))
    plt.figure()
    plt.title("{}-Cross Validation Scores ({} Neighbours)".format(num_folds, num_neighbours))
    plt.plot(scores)
    plt.xlabel("n-trial")
    plt.ylabel("Score (Classification)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    question4()
