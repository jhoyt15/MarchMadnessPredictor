import numpy as np
import csv


"""
Imports the models training data

Args: 
    None
Returns:
    tuple(np.array,np.array): A tuple containing the training data followed by the labels
"""
def importTrainData()->tuple:
    train_data = [] #Training data
    train_labels = [] #Labels of data (Furthest round achieved)
    with open('data/train/KenPom Barttorvik.csv') as kenpomData:
        reader = csv.reader(kenpomData)
        header = reader.__next__()

        #First column is team ID last column is stats
        for row in reader:
            train_data.append([(row[5])]+[row[8]]+row[10::])
            train_labels.append(row[9])
    return (np.array(train_data).transpose(),np.array(train_labels))


"""
Imports the models test data

Args: 
    None
Returns:
    tuple: A tuple containing a matrix containing the test data and the team names
"""
def importTestData()->tuple:
    test_data = [] #Training data
    team_names = []
    with open('data/test/KenPom_Barttorvik_Test.csv') as kenpomData:
        reader = csv.reader(kenpomData)
        header = reader.__next__()

        #First column is team ID last column is stats
        for row in reader:
            test_data.append([(row[5])]+[row[8]]+row[10::])
            team_names.append(row[7])
    return (np.array(test_data),np.array(team_names))
