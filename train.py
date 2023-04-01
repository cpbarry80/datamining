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


def get_meal_data(first=True):

    '''Meal data comprises a 2hr 30 min stretch of CGM data that  starts from tm-30min and extends to tm+2hrs. 
        No meal data comprises 2 hrs of raw data that does not have meal intake. 
        handling missing data. this is a big issue and part of the learning...'''

    if first:
        cgm = pd.read_csv("CGMData.csv",  low_memory=False)
        insulin = pd.read_csv("InsulinData.csv",  low_memory=False)
    else:
        cgm = pd.read_csv("CGM_patient2.csv",  low_memory=False)
        insulin = pd.read_csv("Insulin_patient2.csv", low_memory=False)

    try:
        insulin['dtimestamp'] = pd.to_datetime(insulin['Date'].str.replace(" 00:00:00", "") + ' ' + insulin['Time'],  format='%Y-%m-%d %H:%M:%S')
    except:
        insulin['dtimestamp'] = pd.to_datetime(insulin['Date'].str.replace(" 00:00:00", "") + ' ' + insulin['Time'],  format='%m/%d/%Y %H:%M:%S')

    try:
        cgm['dtimestamp'] = pd.to_datetime(cgm['Date'].str.replace(" 00:00:00", "") + ' ' + cgm['Time'],  format='%Y-%m-%d %H:%M:%S')
    except:
        cgm['dtimestamp'] = pd.to_datetime(cgm['Date'].str.replace(" 00:00:00", "") + ' ' + cgm['Time'],  format='%m/%d/%Y %H:%M:%S')


### https://edstem.org/us/courses/37309/discussion/2851048
### logic for extracting the times specifc to the meal here

    meal_data = insulin.loc[insulin['BWZ Carb Input (grams)'].notna() & insulin['BWZ Carb Input (grams)']>0.0]
    meal_data_copy = meal_data.copy()
    meal_data_copy.loc[:, 'delta'] = meal_data_copy['dtimestamp'].diff().dt.total_seconds().div(60, fill_value=0)
    meal_data_copy = meal_data_copy.loc[meal_data_copy['delta'] < -120]
    
    glucose_data = []
    no_meal_glucose_data = []
    mealtimes = meal_data_copy['dtimestamp'].values
    mealtimes.sort()

    for meal in mealtimes:
        officialstart = meal - pd.Timedelta(minutes=30)  
        end = meal + pd.Timedelta(hours=2)    
        glucose = cgm.loc[(cgm['dtimestamp'] >= officialstart) & (cgm['dtimestamp'] <=end)]['Sensor Glucose (mg/dL)'].values.tolist()
        glucose_data.append(glucose)

        # no meal data
        if meal != mealtimes[0]:
            no_meal = cgm.loc[(cgm['dtimestamp'] <= meal) & (cgm['dtimestamp'] > previous_end)]['Sensor Glucose (mg/dL)'].values.tolist()
            no_meal_glucose_data.append(no_meal[:24])
        previous_end = end

    return pd.DataFrame(glucose_data).iloc[:,0:30], pd.DataFrame(no_meal_glucose_data)



def get_feature_matrix(meal, nans=7):
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


    meal = meal[meal.isnull().sum(axis=1) < nans].interpolate(method='linear',axis=1, limit_direction="both")
    meal = meal[meal.isnull().sum(axis=1) < 1].reset_index(drop=True)

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





meal, fast = get_meal_data()
secondmeal, secondfast = get_meal_data(first=False)

matrix_firstmeal = get_feature_matrix(meal)
matrix_first_no_meal = get_feature_matrix(fast)

matrix_secondmeal = get_feature_matrix(secondmeal)
matrix_second_no_meal = get_feature_matrix(secondfast)

meal_feature_matrix=pd.concat([matrix_firstmeal, matrix_secondmeal]).reset_index().drop(columns='index')
non_meal_feature_matrix=pd.concat([matrix_first_no_meal, matrix_second_no_meal]).reset_index().drop(columns='index')


# validate the machine aka model
# 1. train the model on 80% of the data
# 2. test the model on 20% of the data
#### we need to use some metrics, like accuracy, precision, recall, f1 score, etc. to evaluate the model.

meal_feature_matrix['label']=1
non_meal_feature_matrix['label']=0

data = pd.concat([meal_feature_matrix, non_meal_feature_matrix], ignore_index=True)
data = data.sample(frac=1, random_state=1).reset_index(drop=True)

X = data.drop(columns=['label'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

pipeline = Pipeline([('scaler', StandardScaler()), 
                     ('clf', DecisionTreeClassifier())])

param_grid = {'clf__criterion': ['gini', 'entropy'], 
              'clf__max_depth': [3, 5, 7, None], 
              'clf__min_samples_split': [2, 5, 10], 
              'clf__min_samples_leaf': [1, 2, 4]}

grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=10)
grid_search.fit(X_train, y_train)

# print('Best hyperparameters:', grid_search.best_params_)
# print('Accuracy score:', grid_search.best_score_)

# Evaluate model on test set
accuracy = grid_search.score(X_test, y_test)
# print('Test set accuracy:', accuracy)


with open('grid_search.pickle', 'wb') as f:
    pickle.dump(grid_search, f)





