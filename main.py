import pandas as pd
from auto import get_data
import csv

cgm = pd.read_csv("CGMData.csv")
insulin = pd.read_csv("InsulinData.csv")
# cgm = pd.read_csv('CGMData.csv',low_memory = False, usecols = ['Date','Time','Sensor Glucose (mg/dL)'])
# insulin = pd.read_csv('InsulinData.csv',low_memory = False)


######
cgm['dtimestamp'] = pd.to_datetime(cgm['Date'] + ' ' + cgm['Time'])
date_to_remove = cgm[cgm['Sensor Glucose (mg/dL)'].isna()]['Date'].unique()
cgm = cgm.set_index('Date').drop(index = date_to_remove).reset_index()
insulin['dtimestamp'] = pd.to_datetime(insulin['Date'] + ' ' + insulin['Time'])
start_of_auto_mode = insulin.sort_values(by = 'dtimestamp', ascending = True).loc[insulin['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF'].iloc[0]['dtimestamp']

auto_df = cgm.sort_values(by = 'dtimestamp',ascending = True).loc[cgm['dtimestamp']>=start_of_auto_mode]
manual_df = cgm.sort_values(by = 'dtimestamp',ascending = True).loc[cgm['dtimestamp']<start_of_auto_mode]

####

frames = {"overnight": ['0:00:00','05:59:59'], 
            "daytime": ['6:00:00','23:59:59'], 
            "whole day": ['0:00:00','23:59:59']}

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


