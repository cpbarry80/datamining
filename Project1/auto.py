import pandas as pd
import numpy as np
from datetime import datetime

def get_data(df, frame):

    days = df.groupby('Date')['Sensor Glucose (mg/dL)'].count().index
    df = df.loc[df['Date'].isin(days)]

    # filter relevant days for desired time of day
    format_string = '%H:%M:%S'
    start = datetime.strptime(frame[0], format_string)
    end = datetime.strptime(frame[1], format_string)
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
    starting = (df.loc[(df['Time'] >= start) & (df['Time'] <= end)])

    results = [
        (starting.loc[starting['Sensor Glucose (mg/dL)'] > 180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288 * 100).mean(),
        (starting.loc[starting['Sensor Glucose (mg/dL)'] > 250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288 * 100).mean(),
        (starting.loc[(starting['Sensor Glucose (mg/dL)'] >= 70) & (starting['Sensor Glucose (mg/dL)'] <= 180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288 * 100).mean(),
        (starting.loc[(starting['Sensor Glucose (mg/dL)'] >= 70) & (starting['Sensor Glucose (mg/dL)'] <= 150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288 * 100).mean(),
        (starting.loc[starting['Sensor Glucose (mg/dL)'] < 70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288 * 100).mean(),
        (starting.loc[starting['Sensor Glucose (mg/dL)'] < 54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288 * 100).mean()
        ]

    return [i if not np.isnan(i) else ""for i in results]
