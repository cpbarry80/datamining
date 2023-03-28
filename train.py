import pandas as pd
import numpy as np


cgm = pd.read_csv("CGMData.csv")
insulin = pd.read_csv("InsulinData.csv")

cgm['dtimestamp'] = pd.to_datetime(cgm['Date'] + ' ' + cgm['Time'],  format='%m/%d/%Y %H:%M:%S')
insulin['dtimestamp'] = pd.to_datetime(insulin['Date'] + ' ' + insulin['Time'],  format='%m/%d/%Y %H:%M:%S')

meal_starts = insulin.loc[insulin['BWZ Carb Input (grams)'].notna() & insulin['BWZ Carb Input (grams)']>0.0]
meal_starts_copy = meal_starts.copy()
meal_starts_copy.loc[:, 'delta'] = meal_starts_copy['dtimestamp'].diff().dt.total_seconds().div(60, fill_value=0)
meal_starts = meal_starts.loc[meal_starts['delta'] < -120]
meal_starts['start'] = meal_starts.loc[meal_starts['dtimestamp']- pd.Timedelta(minutes=30)]

meal_starts = insulin.sort_values(by='dtimestamp', ascending=True).loc[insulin['dtimestamp'] >= auto_time]

# Meal data comprises a 2hr 30 min stretch of CGM data that  starts from tm-30min and extends to tm+2hrs. 
# No meal data comprises 2 hrs of raw data that does not have meal intake. 
# handling missing data. this is a big issue and part of the learning...

# no_meal = insulin.loc[insuling['datetime'] - meal_starts < 2hrs]
# no_meal = insulin.loc[insuling['datetime'] - meal_starts < 2hrs]


# missing_data = cgm[cgm['Sensor Glucose (mg/dL)'].isna()]['Date'].unique()


# insulin['dtimestamp'] = pd.to_datetime(insulin['Date'] + ' ' + insulin['Time'])
# auto_time = insulin.loc[insulin['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF'].dtimestamp.values.max()
# auto_df = cgm.sort_values(by = 'dtimestamp', ascending = True).loc[cgm['dtimestamp'] >= auto_time]
