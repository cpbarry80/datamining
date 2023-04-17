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

# https://www.coursera.org/learn/cse572/lecture/sxDM5/project-3-cluster-validation-introductory-video

# Extract features from Meal data
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

    # should this be the summed values of carbs for this meal? 
    mincarb = carbs.min()
    maxcarb = carbs.max()
    if mincarb < min_carbs and mincarb is not np.nan:
        min_carbs = mincarb
    if maxcarb > max_carbs and maxcarb is not np.nan:
        max_carbs = maxcarb
    glucose_data.append(glucose)
    previous_end = end

meal = pd.DataFrame(glucose_data).iloc[:,0:30]
meal = meal[meal.isnull().sum(axis=1) < 7].interpolate(method='linear',axis=1, limit_direction="both")
meal = meal[meal.isnull().sum(axis=1) < 1].reset_index(drop=True)
## end re use of meal matrix from proj 2
binsize = 20
nBins = int((max_carbs - min_carbs)/binsize)

matrix_meal = get_feature_matrix(mealDf)
matrix_nomeal = get_feature_matrix(nomeal)

#for each meal we need a bin number

Meal_features=pd.concat([matrix_meal, matrix_nomeal]).reset_index().drop(columns='index')
# Meal_features=createmealfeaturematrix(mealDf)





pca = PCA(n_components=8)
principalComponents = pca.fit(Meal_features)
PCA_mealdata = pca.fit_transform(Meal_features)

kmeans=KMeans(n_clusters=nBins, random_state=0).fit(PCA_mealdata)
SSE_KMeans=kmeans.inertia_

clusterBinMatrix = [[0 for i in range(nBins)] for i in range(nBins)]
for i in range(kmeans.labels_.shape[0]):
    try:
        clusterBinMatrix[kmeans.labels_[i]][int(mealDf.iloc[i][29])] += 1
    except:
        pass
Kmeans_Entropy =[0,0,0,0,0,0]
Kmeans_Purity = 0
totalPoints = sum(sum(clusterBinMatrix,[]))
for i in range(len(clusterBinMatrix)):
    Kmeans_Purity += (max(clusterBinMatrix[i])*1.0)/(totalPoints*1.0)
    for j in range(len(clusterBinMatrix[i])):
        P = (clusterBinMatrix[i][j]*1.0)/(sum(clusterBinMatrix[i])*1.0)
        if(P != 0):
            Kmeans_Entropy[i] += (-P) * math.log(P,2) * (sum(clusterBinMatrix[i])*1.0/(totalPoints)*1.0)    
Kmeans_Entropy = sum(Kmeans_Entropy)

dbscan = DBSCAN(eps = 210, min_samples = 6).fit(PCA_mealdata) 

for i in range(dbscan.labels_.size):
    if dbscan.labels_[i] == -1:
        min = float('inf')
        l = -1
        for j in range(dbscan.labels_.size):
            if dbscan.labels_[j] != -1:
                eucDist = np.linalg.norm(PCA_mealdata[i] - PCA_mealdata[j])
                if eucDist < min:
                    min = eucDist
                    l = dbscan.labels_[j]
        dbscan.labels_[i] = l            
        
dbscan.labels_ = np.array(dbscan.labels_)

dbScanSSE = 0
for i in range(nBins):
    cluster = PCA_mealdata[dbscan.labels_ == i]
    clusterMean = cluster.mean(axis = 0)
    dbScanSSE += ((cluster - clusterMean) ** 2).sum()

clusterBinMatrix = [[0 for i in range(nBins)] for i in range(nBins)]
for i in range(dbscan.labels_.shape[0]):
    clusterBinMatrix[dbscan.labels_[i]][int(mealDf.iloc[i][29])] += 1
DbScanEntropy = [0,0,0,0,0,0]
DbScanPurity = 0
totalPoints = sum(sum(clusterBinMatrix,[]))
for i in range(len(clusterBinMatrix)):
    DbScanPurity += (max(clusterBinMatrix[i])*1.0)/(totalPoints*1.0)
    for j in range(len(clusterBinMatrix[i])):
        P = (clusterBinMatrix[i][j]*1.0)/(sum(clusterBinMatrix[i])*1.0)
        if(P != 0):
            DbScanEntropy[i] += (-P) * math.log(P,2) * (sum(clusterBinMatrix[i])*1.0/(totalPoints)*1.0)    
DbScanEntropy = sum(DbScanEntropy)

result = [
    [SSE_KMeans,
    dbScanSSE,
    Kmeans_Entropy,
    DbScanEntropy,
    Kmeans_Purity,
    DbScanPurity]
]

resultDf = pandas.DataFrame(result)
resultDf.to_csv('test1Results.csv',float_format='%.6f',index=False, header=False)


# meal_feature_matrix=pd.concat([matrix_firstmeal, matrix_secondmeal]).reset_index().drop(columns='index')
# non_meal_feature_matrix=pd.concat([matrix_first_no_meal, matrix_second_no_meal]).reset_index().drop(columns='index')



# Cluster Meal data based on the amount of carbohydrates in each meal


    #perform clustering
        # Use the features in your Project 2 to cluster the meal data into n clusters. Use DBSCAN and KMeans.
        # Report your accuracy of clustering based on SSE, entropy, and purity metrics.
        # k =N 
        # but dbscan doesnt take clusters as a parameter. how do you get it to N? if they give you 3 compute SSE for each. then arrange
        # them in decreasing order of SSE. 
            # take 1st (max sse) and bisect by kmeans.
            # then it goes from 3 to 4
            # this is just 1 solution. up to you how you want to do it.


    # calculate entropy and purity
        # create matrix



#report results
    # A Result.csv file which contains a 1 X 6 vector. The vector should have the following format:
    # SSE for Kmeans
    # SSE for DBSCAN
    # Entropy for KMeans
    # Entropy for DBSCAN
    # Purity for KMeans
    # Purity for DBSCAN
# df.to_csv('Result.csv',index=False,header=False)


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





