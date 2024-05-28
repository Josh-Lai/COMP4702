#! /usr/bin/env python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def load_data():
    data = pd.read_csv("w3classif.csv", names=["X1", "X2", "Y"])
    X = data.drop(columns=["Y"])
    Y = data["Y"]
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )
    return X_train, X_holdout, y_train, y_holdout


def question_1(X_train, X_holdout, y_train, y_holdout):
    # Run the training on the training set then test on the hold out dataset
    dec_tree = DecisionTreeClassifier()
    dec_tree.fit(X_train, y_train)
    E_train = accuracy_score(y_train, dec_tree.predict(X_train))
    E_hold_out = accuracy_score(y_holdout, dec_tree.predict(X_holdout))
    print("Etrain:", E_train)
    print("Ehold-out:", E_hold_out)


"""
Train a Random Forest Classifier, this method uses x amount of trees and split datasets into bagging
trees 
"""


def question_2(X_train, X_holdout, y_train, y_holdout):
    forest_tree = RandomForestClassifier()
    forest_tree.fit(X_train, y_train)
    y_train_pred = forest_tree.predict(X_train)
    y_holdout_pred = forest_tree.predict(X_holdout)

    E_train = accuracy_score(y_train, y_train_pred)
    E_holdout = accuracy_score(y_holdout, y_holdout_pred)
    print("Random Forest Scores")
    print("Etrain:", E_train)
    print("Ehold-out:", E_holdout)


"""
This prac aims at learning Ensemble methods. These are methods aimed at improving the accuracy by combining predictors together
"""
if __name__ == "__main__":
    X_train, X_holdout, y_train, y_holdout = load_data()
    question_1(X_train, X_holdout, y_train, y_holdout)
    question_2(X_train, X_holdout, y_train, y_holdout)
    """
    The Random Forest classifier has a higher hold-out accuracy
    This is expected as bagged methods use several portions of the dataset that is sampled
    with replacement to obtain a more accurate estimation of the data trends of the data set
    """
