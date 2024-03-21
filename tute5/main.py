#!/usr/bin/env python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def create_train_test_data(data_size=10,test_size=0.3):
    data = pd.read_csv('w3classif.csv', names=["X1", "X2", "Y"])
    X = data.drop("Y", axis=1)
    y = data["Y"]

    X_trains, X_tests = [], []
    y_trains, y_tests = [], []
    for i in range(0, data_size):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size)
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
    return X_trains, X_tests, y_trains, y_tests

def question2():
    # Read the training and test sets, train a KNN classifier for them
    x_trains, x_tests, y_trains, y_tests = create_train_test_data()
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
    print("Test misclassification Average {}%".format(test_acc_avg))
    print("Training misclassification Average {}%".format(train_acc_avg))


if __name__ == "__main__":
    question2()
