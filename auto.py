import pandas as pd
import numpy as np


def get_data(df1, frame):
    df = df1.copy()
    df = df.set_index('dtimestamp')
    list1 = df.groupby('Date')['Sensor Glucose (mg/dL)'].count().where(lambda x:x>0.8*288).dropna().index.tolist() 
    df = df.loc[df['Date'].isin(list1)]

    starting = df.between_time(frame[0],frame[1])[['Date','Time','Sensor Glucose (mg/dL)']]

    results = [
        (starting.loc[df['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100).mean(),
        (starting.loc[df['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100).mean(),
        (starting.loc[(df['Sensor Glucose (mg/dL)']>=70) & (df['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100).mean(),
        (starting.loc[(df['Sensor Glucose (mg/dL)']>=70) & (df['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100).mean(),
        (starting.loc[df['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100).mean(),
        (starting.loc[df['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100).mean()
        ]

    return [i for i in results]