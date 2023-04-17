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


def get_feature_matrix(meal):
    '''
    args: nans: number of nans allowed in a row
    
    FEATURES 
    https://www.coursera.org/learn/cse572/lecture/MPLNQ/project-2-machine-model-training-introductory-video-2
    1. time difference between when the meal was taken versus when the CGM reached maximum
    2. ( CGM max (aka Dg) - CGM meal ) / CGM meal   
    3. fast fourier transform....in this case, what we'll have is you will have the power at.... So you have four different numbers.
        a. poweruency F1
        b. F1 
        c. power at poweruency,F2 
        d. F2
    4. slope of cgm right at meal time is very important feature. just subtract 2 consecutive data points
    5. lets say after that you get a difference, you take another difference. the double difference is proprotional to the meal amount'''


    power_f1=[]
    f1=[]
    power_f2=[]
    f2=[]

    diff=[]
    dubdiff=[]

    for i in range(len(meal)):
        frontierft = abs(fft(meal.iloc[i].values)).tolist()
        sort_fft = frontierft.copy()
        sort_fft.sort()

        f1power = sort_fft[-2] #skip the first one
        f1location = frontierft.index(f1power)
        f2power = sort_fft[-3]
        f2location = frontierft.index(f2power)

        power_f1.append(f1power)
        power_f2.append(f2power)
        f1.append(f1location)
        f2.append(f2location)

        diff.append(np.diff(meal.iloc[:, 9:11].iloc[i])[0])
        dubdiff.append(np.diff(meal.iloc[:, 9:11].iloc[i])[0]**2)


    matrix = pd.DataFrame()
    matrix['tau'] = (meal.idxmin(axis=1) - meal.idxmax(axis=1)) * 5
    matrix['glucose_diff_normalized'] = (meal.max(axis=1) - meal.min(axis=1)) / (meal.min(axis=1))
    
    
    matrix['power_f1'] = power_f1
    matrix['power_f2'] = power_f2
    matrix['f1'] = f1
    matrix['f2'] = f2
    matrix['diffa'] = diff
    matrix['diffb'] = dubdiff

    return matrix

