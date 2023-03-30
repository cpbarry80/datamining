import pandas as pd
import numpy as np
from scipy.fftpack import fft, ifft,rfft




def get_meal_data(first=True):

    '''Meal data comprises a 2hr 30 min stretch of CGM data that  starts from tm-30min and extends to tm+2hrs. 
        No meal data comprises 2 hrs of raw data that does not have meal intake. 
        handling missing data. this is a big issue and part of the learning...'''

    if first:
        cgm = pd.read_csv("CGMData.csv")
        insulin = pd.read_csv("InsulinData.csv")
    else:
        cgm = pd.read_csv("CGM_patient2.csv")
        insulin = pd.read_csv("Insulin_patient2.csv")

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



def get_feature_matrix(meal, nans=7, s1=9, e1=24, s2=20, e2=23, fulllength=30):
    '''
    args: nans: number of nans allowed in a row
    s1: start of meal
    e1: end of meal
    s2: start of fast
    e2: end of fast
    
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


    maximum = meal.iloc[:,s1:e1].idxmax(axis=1)
    tm = meal.iloc[:,s2:e2].idxmin(axis=1)

    diff=[]
    diff2=[]


    for i in range(len(meal)):
        fft = abs(rfft(meal.iloc[i].values))
        sort_fft = fft.copy()
        sort_fft.sort(reverse=True)

        f1power = sort_fft[1] #skip the first one
        f1location = fft.index(f1power)
        f2power = sort_fft[2]
        f2location = fft.index(f2power)

        power_f1.append(f1power)
        power_f2.append(f2power)
        f1.append(f1location)
        f2.append(f2location)

        diff.append(np.diff(meal.iloc[:,int(maximum[i]):int(tm[i])].iloc[i].tolist()).max())
        diff2.append(np.diff(np.diff(meal.iloc[:,int(maximum[i]):int(tm[i])].iloc[i].tolist())).max())

    matrix=pd.DataFrame()
    matrix['tau'] = (meal.iloc[:,s1:e2].idxmin(axis=1) - meal.iloc[:,s1:e1].idxmax(axis=1)) * 5
    #cgm max - cgm min
    matrix['glucose_diff_normalized'] = (meal.iloc[:,s1:e1].max(axis=1) - meal.iloc[:,s2:e2].min(axis=1)) / (meal.iloc[:,s2:e2].min(axis=1))
    
    
    matrix['power_f1'] = power_f1
    matrix['power_f2'] = power_f2
    matrix['f1'] = f1
    matrix['f2'] = f2
    matrix['1stDifferential'] = diff
    matrix['2ndDifferential'] = diff2

    return matrix





meal, fast = get_meal_data()
secondmeal, secondfast = get_meal_data(first=False)

matrix_firstmeal = get_feature_matrix(meal)
matrix_first_no_meal = get_feature_matrix(fast, s1=0, e1=19, s2=0, e2=24, fulllength=24)

matrix_secondmeal = get_feature_matrix(secondmeal, fulllength=24)
matrix_secondfast = get_feature_matrix(secondfast, s1=0, e1=19, s2=0, e2=24, fulllength=24)





# validate the machine aka model
# 1. train the model on 80% of the data
# 2. test the model on 20% of the data
#### we need to use some metrics, like accuracy, precision, recall, f1 score, etc. to evaluate the model.


# test.py
# reads the model and test data and saves to results.csv
