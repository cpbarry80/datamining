import pandas as pd
import csv
import numpy as np
import pickle

# # test.csv which has the N x 24 matrix 
new_data = pd.read_csv("test.csv", header=None)

with open('grid_search.pickle', 'rb') as f:
    grid_search = pickle.load(f)

# # Result.csv file which has N x 1 vector of 1s and 0s, where 1 denotes meal, 0 denotes no meal.
predictions = grid_search.best_estimator_.predict(new_data)

pd.DataFrame(predictions).to_csv('Result.csv',index=False,header=False)



