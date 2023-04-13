from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score

# create the data as a pandas dataframe
data = pd.DataFrame({
    'X': [1, 2, 2, 8, 8, 25],
    'Y': [2, 2, 3, 7, 8, 80],
    'Sample': ['A', 'B', 'C', 'D', 'E', 'F']
})

# create a numpy array from the X and Y columns
X = np.array(data[['X', 'Y']])

# perform DBSCAN clustering
dbscan = DBSCAN(eps=3, min_samples=2)
dbscan.fit(X)

# count the number of core samples
core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
n_core_samples = len(set(dbscan.labels_[core_samples_mask]))

# print the number of core samples
print("Number of core samples:", n_core_samples)


# compute the Silhouette coefficient for each sample
silhouette_vals = silhouette_samples(X, dbscan.labels_)

# compute the average Silhouette coefficient
silhouette_avg = silhouette_score(X, dbscan.labels_)

# print the Silhouette coefficient for each sample
for i, s in enumerate(data['Sample']):
    print("Sample", s, "has Silhouette coefficient:", silhouette_vals[i])

# print the average Silhouette coefficient
print("Average Silhouette coefficient:", silhouette_avg)
