import pandas as pd
import openpyxl
from datetime import datetime, timedelta
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

def integer2date(s):
    s = str(s)
    return  datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]), hour=int(s[8:]))

def adding_hours(h):
    return timedelta(hours = h)

def day_func(ts):
    return ts.day

def hr_func(ts):
    return ts.hour

def month_func(ts):
    return ts.month

def date_conversion(df):
    """ Converts the given time of forecast and distance to the forecast into the forecast date."""
    df['forecast_time'] = df.date.apply(lambda x: integer2date(x))
    df['hours_added'] = df.hors.apply(lambda x: adding_hours(x))
    df['date'] = df['forecast_time'] + df['hours_added']
    df = df.drop(['hours_added', 'hors'], axis = 1)
    return df

def display_scores(scores):
    print('Scores:', scores)
    print('Mean:', np.mean(scores))
    print('Std:', np.std(scores))
    
    
def write_results(path, sheet, df):
    """
    Writes a given dataframe into an excel file
    :param dicts:
    """
    writer = pd.ExcelWriter(path, engine="openpyxl")
    if os.path.exists(path):
        book = openpyxl.load_workbook(path)
        writer.book = book

    df.to_excel(writer, sheet, index=True)

    writer.save()
    writer.close()