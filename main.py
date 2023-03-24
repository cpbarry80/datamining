import pandas as pd
from auto import get_data
import csv
import numpy as np

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


