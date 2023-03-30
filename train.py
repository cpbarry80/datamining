import pandas as pd
import numpy as np




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





def get_feature_matrix(meal):
    '''FEATURES 
    https://www.coursera.org/learn/cse572/lecture/MPLNQ/project-2-machine-model-training-introductory-video-2
    1. time difference between when the meal was taken versus when the CGM reached maximum
    2. ( CGM max (aka Dg) - CGM meal ) / CGM meal   
    3. fast fourier transform....in this case, what we'll have is you will have the power at frequency F1 and F1 and you'll have the power at frequency,F2 and F2. So you have four different numbers.
    4. slope of cgm right at meal time is very important feature. just subtract 2 diff data points
    5. lets say after that you get a difference, you take another difference. the double difference is proprotional to the meal amount'''

    meal = meal.values
    fast = fast.values
    feature_matrix = np.zeros((len(meal)+len(fast), 4))
    for i in range(len(meal)):
        feature_matrix[i, 0] = np.max(meal[i, :]) - meal[i, 0]
        feature_matrix[i, 1] = (np.max(meal[i, :]) - meal[i, 0])/meal[i, 0]
        feature_matrix[i, 2] = np.max(meal[i, :]) - np.min(meal[i, :])
        feature_matrix[i, 3] = meal[i, 1] - meal[i, 0]
    for i in range(len(fast)):
        feature_matrix[i+len(meal), 0] = np.max(fast[i, :]) - fast[i, 0]
        feature_matrix[i+len(meal), 1] = (np.max(fast[i, :]) - fast[i, 0])/fast[i, 0]
        feature_matrix[i+len(meal), 2] = np.max(fast[i, :]) - np.min(fast[i, :])
        feature_matrix[i+len(meal), 3] = fast[i, 1] - fast[i, 0]
    return feature_matrix





meal, fast = get_meal_data()
secondmeal, secondfast = get_meal_data(first=False)

# validate the machine aka model
# 1. train the model on 80% of the data
# 2. test the model on 20% of the data
#### we need to use some metrics, like accuracy, precision, recall, f1 score, etc. to evaluate the model.


# test.py
# reads the model and test data and saves to results.csv
