import pandas as pd
import numpy as np


def get_meal_data(first=True):
    if first:
        cgm = pd.read_csv("CGMData.csv")
        insulin = pd.read_csv("InsulinData.csv")
    else:
        cgm = pd.read_csv("CGM_patient2.csv")
        insulin = pd.read_csv("Insulin_patient2.csv")

    insulin['dtimestamp'] = pd.to_datetime(insulin['Date'] + ' ' + insulin['Time'],  format='%m/%d/%Y %H:%M:%S')
    cgm['dtimestamp'] = pd.to_datetime(cgm['Date'] + ' ' + cgm['Time'],  format='%m/%d/%Y %H:%M:%S')
    meal_data = insulin.loc[insulin['BWZ Carb Input (grams)'].notna() & insulin['BWZ Carb Input (grams)']>0.0]
    meal_data_copy = meal_data.copy()
    meal_data_copy.loc[:, 'delta'] = meal_data_copy['dtimestamp'].diff().dt.total_seconds().div(60, fill_value=0)
    meal_data_copy = meal_data_copy.loc[meal_data_copy['delta'] < -120]
    meal_data_copy['start'] = meal_data_copy['dtimestamp']- pd.Timedelta(minutes=30)
    
    glucose_data = []
    for start in meal_data_copy['start'].values:
        end = start + pd.Timedelta(hours=2, minutes=30)    
        glucose = cgm.loc[(cgm['dtimestamp'] >= start) & (cgm['dtimestamp'] <=end)]['Sensor Glucose (mg/dL)'].values.tolist()
        glucose_data.append(glucose)

    return pd.DataFrame(glucose_data)



firstmeal = get_meal_data()
secondmeal = get_meal_data(first=False)



# Meal data comprises a 2hr 30 min stretch of CGM data that  starts from tm-30min and extends to tm+2hrs. 
# No meal data comprises 2 hrs of raw data that does not have meal intake. 
# handling missing data. this is a big issue and part of the learning...

# no_meal = insulin.loc[insuling['datetime'] - meal_starts < 2hrs]
# no_meal = insulin.loc[insuling['datetime'] - meal_starts < 2hrs]


# missing_data = cgm[cgm['Sensor Glucose (mg/dL)'].isna()]['Date'].unique()


# insulin['dtimestamp'] = pd.to_datetime(insulin['Date'] + ' ' + insulin['Time'])
# auto_time = insulin.loc[insulin['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF'].dtimestamp.values.max()
# auto_insulin = cgm.sort_values(by = 'dtimestamp', ascending = True).loc[cgm['dtimestamp'] >= auto_time]


#FEATURES 
# https://www.coursera.org/learn/cse572/lecture/MPLNQ/project-2-machine-model-training-introductory-video-2
# 1. time difference between when the meal was taken versus when the CGM reached maximum
# 2. ( CGM max (aka Dg) - CGM meal ) / CGM meal   
# 3. fast fourier transform....in this case, what we'll have is you will have the power at frequency F1 and F1 and you'll have the power at frequency,F2 and F2. So you have four different numbers.
# 4. slope of cgm right at meal time is very important feature. just subtract 2 diff data points
# 5. lets say after that you get a difference, you take another difference. the double difference is proprotional to the meal amount


# validate the machine aka model
# 1. train the model on 80% of the data
# 2. test the model on 20% of the data
#### we need to use some metrics, like accuracy, precision, recall, f1 score, etc. to evaluate the model.


# test.py
# reads the model and test data and saves to results.csv
