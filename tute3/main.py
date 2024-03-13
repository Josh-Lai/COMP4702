#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

"""
    Demo taken from https://github.com/nathasha-naranpanawa/COMP4702_2024/blob/main/PracW3.ipynb
"""


def question4():
    # QUestion 4 Regression model
    # The main difference between regression and classification is that the
    # regression model makes use of numerical values

    data_reg = pd.read_csv("w3regr.csv", names=["X", "Y"])
    X = data_reg.drop("Y", axis=1)
    y = data_reg["Y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    plt.scatter(X, y)
    plt.show()
    k = 3
    knr = KNeighborsRegressor(n_neighbors=k)
    knr.fit(X_train, y_train)

    y_train_predict = knr.predict(X_train)
    y_test_predict = knr.predict(X_test)

    # Plot the training data:
    fig, ax = plt.subplots()

    ax.plot(X, y, label='Training Data')

    ax.scatter(X_train,y_train_predict, c= 'g', label='Predicted Function_train')
    ax.scatter(X_test,y_test_predict, c ='r', label='Predicted Function_test')

    plt.legend()
    plt.show()



def question5():
    data = pd.read_csv("w3classif.csv", names=["X1", "X2", "Y"])

    # Question Randomly shuffle using sklearn
    X = data.drop("Y", axis=1)
    y = data["Y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Create tree
    d = 3
    tree = DecisionTreeClassifier(criterion="gini", max_depth=d)
    tree.fit(X_train, y_train)
    # Predict
    y_train_predict = tree.predict(X_train)
    y_test_predict = tree.predict(X_test)

    # Estimate the classification error
    train_misclass = 1 - accuracy_score(y_train_predict, y_train)
    test_misclass = 1 - accuracy_score(y_test_predict, y_test)
    print("Train Misclassification {}".format(train_misclass * 100))
    print("Test Misclassication {}".format(test_misclass * 100))
    # Display the decision boundaries
    X1_train = X_train.loc[:, "X1"]
    X2_train = X_train.loc[:, "X2"]

# Generate a meshgrid of points to cover the feature space
    h = 0.02
    x_min, x_max = X1_train.min() - 0.5, X1_train.max() + 0.5
    y_min, y_max = X2_train.min() - 0.5, X2_train.max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

# Predict class labels for the points in the meshgrid
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

# Plot the decision regions
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

# Plot the training data points
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    plt.scatter(X1_train, X2_train, c=y_train,
                cmap=cmap_bold, edgecolor='k', s=20)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Decision Tree Boundaries (max depth = %d)' % d)
    plt.show()

def question6():
    #Using a decision tree regressor
    # This method is identical to the classifier, execpt this time, the regions contain the 
    # average of the points described in the region
    data_reg = pd.read_csv("w3regr.csv", names=["X", "Y"])
    X = data_reg.drop("Y", axis=1)
    y = data_reg["Y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    d = 3
    tree = DecisionTreeRegressor(criterion="squared_error", max_depth=d)
    tree.fit(X_train, y_train)

    # Obtain prediction information
    y_train_predict = tree.predict(X_train)
    y_test_predict = tree.predict(X_test)

    # Evaluate the MSE between the two sets
    train_acc = mean_squared_error(y_train_predict, y_train)
    test_acc = mean_squared_error(y_test_predict, y_test)

    print("Test Accuracy MSE: {}".format(test_acc))
    print("Train Accuracy MSE: {}".format(train_acc))
    # Plot the results
    fig, ax = plt.subplots()

    ax.plot(X, y, label='Training Data')

    ax.scatter(X_train,y_train_predict, c= 'g', label='Predicted Function_train')
    ax.scatter(X_test,y_test_predict, c ='r', label='Predicted Function_test')
    plt.title("Decision Tree Regressor (max depth = {}) (Testing MSE = {})".format(d, test_acc))

    plt.legend()
    plt.show()
    
    return




def question123():
    data = pd.read_csv("w3classif.csv", names=["X1", "X2", "Y"])
    fig = plt.figure()
    colours = ListedColormap(['r', 'b'])
    scatter = plt.scatter(data.loc[:, "X1"], data.loc[:, "X2"],
                          c=data.loc[:, "Y"], cmap=colours, label=data.loc[:, "Y"])
    plt.legend(*scatter.legend_elements())
    plt.ylabel('X1')
    plt.xlabel('X2')
    plt.show()

    # Question Randomly shuffle using sklearn
    X = data.drop("Y", axis=1)
    y = data["Y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Question 3
    print("QUESTION 3")
    print("part a.)")
    # sklearn has a KNeighborsClassifier use this
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    # Predict
    y_predict = knn.predict(X_test)

    # Evaulate accuracy
    print("Testing Misclassification Rate {}%".format(
        (1 - knn.score(X_test, y_test)) * 100))
    print("Training Misclassication Rate {}%".format(
        (1 - knn.score(X_train, y_train)) * 100))

    print("part b.)")

    X1_train = X_train.loc[:, "X1"]
    X2_train = X_train.loc[:, "X2"]

# Generate a meshgrid of points to cover the feature space
    h = 0.02
    x_min, x_max = X1_train.min() - 0.5, X1_train.max() + 0.5
    y_min, y_max = X2_train.min() - 0.5, X2_train.max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

# Predict class labels for the points in the meshgrid
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

# Plot the decision regions
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

# Plot the training data points
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    plt.scatter(X1_train, X2_train, c=y_train,
                cmap=cmap_bold, edgecolor='k', s=20)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('k-NN decision regions (k = %d)' % k)
    # plt.show()
    # As The k neighbors increases, the boundaries become more detailed, and the system becomes
    # overfitted


if __name__ == "__main__":
    question6()
