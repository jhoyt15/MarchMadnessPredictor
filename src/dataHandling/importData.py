import numpy as np
import csv

def importData()->np.array:
    train_data = []
    with open('data/train/KenPom Barttorvik.csv') as kenpomData:
        reader = csv.reader(kenpomData)
        header = reader.__next__()
        print(header)

        #First column is team ID last column is stats
        for row in reader:
            train_data.append([row[5]]+row[8::])
    return np.array(train_data).transpose()
