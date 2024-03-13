import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def KNN_classification(trainingData: pd.DataFrame, testingData: pd.DataFrame, k: int):
    # For each of the testing sets:
    #   Loop over the training data and find the norm
    #   For the k nearest neighbours, calssify with a majority vote

    t = testingData.shape[0]
    n = trainingData.shape[0]
    predictionData = testingData
    print("Running Classifier with {} NN".format(k))
    correct = 0
    for i in range(0, t):
        x_star = testingData.iloc[int(i)]
        dsts = []
        for j in range(0, n):
            x = trainingData.iloc[int(j)]
            # Compute the normals and append to list
            # squaredNormal = (x1* - x1)2 + (0][x2* - x2)
            squaredNormal = (x_star["X1"] - x["X1"])**2 + \
                (x_star["X2"] - x["X2"])**2
            dsts.append(squaredNormal)
        nearestDsts = sorted(dsts)[:k]
        # Find corresponding neighbours
        nearestNeighbours = [dsts.index(num) for num in nearestDsts]
        # find the outputs then maxVote this
        nearestTraining = trainingData.loc[nearestNeighbours, :]
        commonVote = nearestTraining["Y"].value_counts().idxmax()
        predictionData.at[int(i), "Y"] = commonVote
        if (commonVote == x_star["Y"]):
            correct += 1

    print("Found {} correct".format(correct))
    print("Misclassficiation rate {}%".format((1 - correct/t) * 100))


if __name__ == "__main__":
    TRAINING_FRAC = 0.7

    # Import the data
    data = pd.read_csv("w3classif.csv", names=["X1", "X2", "Y"])
    # Plot the data as per the TUte instructions
    colours = ListedColormap(['r', 'b'])
    scatter = plt.scatter(data.loc[:, "X1"], data.loc[:, "X2"],
                          c=data.loc[:, "Y"], cmap=colours, label=data.loc[:, "Y"])
    plt.legend(*scatter.legend_elements())
    plt.ylabel('X1')
    plt.xlabel('X2')
    # plt.show()
    # print(data)
    """
        Question 2
    """
    # Part a randomly shuffle the data sets and split into different parts
    numRows = data.shape[0]
    print("Read a a total of {} rows".format(numRows))
    trainingSize = int(TRAINING_FRAC * numRows)
    shuffledData = data.sample(frac=1)
    trainingData = shuffledData.iloc[0:trainingSize].reset_index(drop=True)
    testingData = shuffledData.iloc[trainingSize:numRows].reset_index(
        drop=True)

    """ 
        Question 3 
    """
    # part a build a k-NN classifier for the data sets
    # In this problem, n = 70%, Do a knn boundaries
    KNN_classification(trainingData, testingData, 3)
