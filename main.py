from matrixgenerator import get_meal_data, get_feature_matrix


# https://www.coursera.org/learn/cse572/lecture/sxDM5/project-3-cluster-validation-introductory-video

# Extract features from Meal data


    #load data

    #extract ground truth
        # Derive the max and min value of meal intake amount from the Y column of the Insulin data. Discretize
        # the meal amount in bins of size 20. Consider each row in the meal data matrix that you generated in
        # Project 2. Put them in the respective bins according to their meal amount label.
        # In total, you should have n = (max-min)/20 bins.

meal, nomeal = get_meal_data()

meal_feature_matrix = get_feature_matrix(meal)
no_meal_feature_matrix = get_feature_matrix(nomeal)


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





