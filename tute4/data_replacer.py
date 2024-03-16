#!/usr/bin/env python

import pandas as pd
import numpy as np

def remove_nan(df : pd.DataFrame):
    colCount = 0
    rowsToDrop = set()
    for column in df:
        print(column)
        #print(df[column])
        nanCount =0
        for num, el in enumerate(df[column]):
            if (pd.isna(el)):
                rowsToDrop.add(num)
                nanCount+=1
        print(f"Found {nanCount} nans\n")
        print(rowsToDrop)
        colCount += 1
    print(f"{colCount} columns")
    df = df.drop(rowsToDrop)
    return df

def replace_with_mean(df : pd.DataFrame):
    for column in df:
        # If the datatype of the column is a string use median
        # Otherwise use the average
        isNumeric = pd.api.types.is_numeric_dtype(df[column].dtype)
        nanReplacement = 0
        if (not isNumeric):
            # Use Counter for the system
            ElCount = Counter(df[column])
            mostCommon = ElCount.most_common()[0][0]
            if (pd.isna(mostCommon)):
                mostCommon = ElCount.most_common()[1][0]
            print(column, " Most Common", mostCommon," ", df[column].dtype)
            nanReplacement = mostCommon
        else:
            nanReplacement = df[column].mean()
        for num, el in enumerate(df[column]):
            if (pd.isna(el)):
                # Replace this element with the mean
                df.at[num, column] = nanReplacement
    print("Replaced Empty Data sets with mean")
    return df

