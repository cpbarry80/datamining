import pandas as pd
import csv
import numpy as np
import pickle

# test.csv which has the N x 24 matrix 
testdata = pd.read_csv("test.csv")

grid_search = pickle.load('grid_search.pickle')

df = grid_search(testdata)

df.to_csv("Result.csv")
# Result.csv file which has N x 1 vector of 1s and 0s, where 1 denotes meal, 0 denotes no meal.



