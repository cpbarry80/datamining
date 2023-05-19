import pandas as pd
from auto import get_data
import csv
import numpy as np


# Extract feature data from a data set.
# ● Synchronize data from two sensors.
# ● Compute and report overall statistical measures from data.


cgm = pd.read_csv("CGMData.csv")
insulin = pd.read_csv("InsulinData.csv")

#clean CGM
cgm['dtimestamp'] = pd.to_datetime(cgm['Date'] + ' ' + cgm['Time'],  format='%m/%d/%Y %H:%M:%S')
missing_data = cgm[cgm['Sensor Glucose (mg/dL)'].isna()]['Date'].unique()
cgm = cgm.loc[~cgm['Date'].isin(list(missing_data))]

insulin['dtimestamp'] = pd.to_datetime(insulin['Date'] + ' ' + insulin['Time'])

auto_time = insulin.loc[insulin['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF'].dtimestamp.values.max()

auto_df = cgm.sort_values(by = 'dtimestamp', ascending = True).loc[cgm['dtimestamp'] >= auto_time]
manual_df = cgm.sort_values(by = 'dtimestamp', ascending = True).loc[cgm['dtimestamp'] < auto_time]


frames = {"night": ['0:00:00','05:59:59'], 
            "day": ['6:00:00','23:59:59'], 
            "wholeday": ['0:00:00','23:59:59']}

modes = ["Manual", "Auto"]

csv_data = []
for mode in modes:
    total_mode = []
    for frame, defintion in frames.items():
        if mode == "Manual":
            mode_time = get_data(manual_df, defintion)
        else:
            mode_time = get_data(auto_df, defintion)
        [total_mode.append(i) for i in mode_time]
    csv_data.append(total_mode)



file = open('Result.csv', 'w', newline ='')
 
with file:   
    write = csv.writer(file)
    write.writerows(csv_data)

# 1. Percentage time in hyperglycemia (CGM > 180 mg/dL),
# 2. percentage of time in hyperglycemia critical (CGM > 250 mg/dL),
# 3. percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL),
# 4. percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL),
# 5. percentage time in hypoglycemia level 1 (CGM < 70 mg/dL), and
# 6. percentage time in hypoglycemia level 2 (CGM < 54 mg/dL).

# Each of the above-mentioned metrics are extracted in three different time intervals: daytime (6 am to
# midnight), overnight (midnight to 6 am), and whole day (12 am to 12 am

# The metrics will be computed for two cases:
# ● Case A: Manual mode
# ● Case B: Auto mode
