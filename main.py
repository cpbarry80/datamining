from matrixgenerator import get_feature_matrix
import pandas as pd
import numpy as np
from scipy.fftpack import fft, ifft, rfft
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
from numpy.fft import fft
from sklearn.decomposition import PCA
import math
from sklearn.cluster import KMeans,DBSCAN
from sklearn.metrics.cluster import entropy
import numpy as np
from scipy.stats import entropy
import numpy as np
from sklearn import metrics
# https://www.coursera.org/learn/cse572/lecture/sxDM5/project-3-cluster-validation-introductory-video

# Extract features from Meal dataff
    #extract ground truth
        # Derive the max and min value of meal intake amount from the Y column of the Insulin data. Discretize
        # the meal amount in bins of size 20. Consider each row in the meal data matrix that you generated in
        # Project 2. Put them in the respective bins according to their meal amount label.
        # In total, you should have n = (max-min)/20 bins.

cgm = pd.read_csv("CGMData.csv",  low_memory=False)
insulin = pd.read_csv("InsulinData.csv",  low_memory=False)

try:
    insulin['dtimestamp'] = pd.to_datetime(insulin['Date'].str.replace(" 00:00:00", "") + ' ' + insulin['Time'],  format='%Y-%m-%d %H:%M:%S')
except:
    insulin['dtimestamp'] = pd.to_datetime(insulin['Date'].str.replace(" 00:00:00", "") + ' ' + insulin['Time'],  format='%m/%d/%Y %H:%M:%S')

try:
    cgm['dtimestamp'] = pd.to_datetime(cgm['Date'].str.replace(" 00:00:00", "") + ' ' + cgm['Time'],  format='%Y-%m-%d %H:%M:%S')
except:
    cgm['dtimestamp'] = pd.to_datetime(cgm['Date'].str.replace(" 00:00:00", "") + ' ' + cgm['Time'],  format='%m/%d/%Y %H:%M:%S')

meal_data = insulin.loc[insulin['BWZ Carb Input (grams)'].notna() & insulin['BWZ Carb Input (grams)']>0.0]
meal_data_copy = meal_data.copy()
meal_data_copy.loc[:, 'delta'] = meal_data_copy['dtimestamp'].diff().dt.total_seconds().div(60, fill_value=0)
meal_data_copy = meal_data_copy.loc[meal_data_copy['delta'] < -120]

glucose_data = []
mealtimes = meal_data_copy['dtimestamp'].values
mealtimes.sort()

min_carbs = 0
max_carbs = 0
for meal in mealtimes:
    officialstart = meal - pd.Timedelta(minutes=30)  
    end = meal + pd.Timedelta(hours=2)    
    glucose = cgm.loc[(cgm['dtimestamp'] >= officialstart) & (cgm['dtimestamp'] <=end)]['Sensor Glucose (mg/dL)'].values.tolist()
    carbs = insulin.loc[(insulin['dtimestamp'] >= officialstart) & (insulin['dtimestamp'] <=end)]['BWZ Carb Input (grams)']

    mincarb = carbs.min()
    maxcarb = carbs.max()
    if mincarb < min_carbs and mincarb is not np.nan:
        min_carbs = mincarb
    if maxcarb > max_carbs and maxcarb is not np.nan:
        max_carbs = maxcarb
    glucose_data.append(glucose)
    previous_end = end

# assign bins to each meal
meal_data_copy['carb_bin'] = (meal_data_copy['BWZ Carb Input (grams)'] / (min_carbs + 20)).astype(int)

meal = pd.DataFrame(glucose_data).iloc[:,0:30]

meal_data_copy.reset_index(inplace=True)
meal_data_copy.drop(meal[meal.isnull().sum(axis=1) >= 7].index, inplace=True, axis=0)

meal = meal[meal.isnull().sum(axis=1) < 7].interpolate(method='linear',axis=1, limit_direction="both")
meal = meal[meal.isnull().sum(axis=1) < 1].reset_index(drop=True)
## end re use of meal matrix from proj 2


#perform clustering
    # Use the features in your Project 2 to cluster the meal data into n clusters. Use DBSCAN and KMeans.
    # Report your accuracy of clustering based on SSE, entropy, and purity metrics.
    # k =N 
    # but dbscan doesnt take clusters as a parameter. how do you get it to N? if they give you 3 compute SSE for each. then arrange
    # them in decreasing order of SSE. 
        # take 1st (max sse) and bisect by kmeans.
        # then it goes from 3 to 4
        # this is just 1 solution. up to you how you want to do it.


meal_features = get_feature_matrix(meal)

meal_features_scaled = StandardScaler().fit_transform(meal_features)
kmeans = KMeans(init="random", n_clusters=6, n_init=10, random_state=1)
kmeans.fit(meal_features_scaled)
kmeansse=kmeans.inertia_

true_labels = meal_data_copy['carb_bin'].astype(int).to_list()
cluster_labels = list(kmeans.labels_.astype(int))

value,counts = np.unique(true_labels, return_counts=True)
kmeansentropy = entropy(value, base=2)

contingency_matrix = metrics.cluster.contingency_matrix(true_labels, cluster_labels)
kmeanspurity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 


# aim is to get N clusters
# dcscan can give you any number of clusters
dbScanSSE = 0
DbScanEntropy = 0
DbScanPurity = 0

df = pd.DataFrame([[kmeansse, dbScanSSE, kmeansentropy, DbScanEntropy, kmeanspurity, DbScanPurity]])
print(df)
# df.to_csv('Result.csv', index=False, header=False)

# The autograder in Coursera will evaluate your code based on the following criteria:
# ● 50 points for developing a code in Python that takes the dataset and performs clustering.
# ● 20 points for developing a code in Python that implements a function to compute SSE, entropy
# and purity metrics. These two can be written in the same file.
# ● 30 points will be evaluated on the supervised cluster validation results obtained by your code.

# Note: The autograder has fixed values for minimumEntropy, maximumPurity, and standard deviation for K-Means and DBSCAN and uses these values to perform a few mathematical calculations. Your K-Means and DBSCAN Purity and Entropy should be in the range of these calculations. Below are the minEntropy and maxPurity values:
# Minimum K-Means Entropy: 0.3235
# Minimum DBSCAN Entropy: 0.1739
# Maximum K-Means Purity: 0.875
# Maximum DBSCAN Purity: 1





