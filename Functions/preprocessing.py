from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from Functions.helper_functions import *


def feature_engineering(df):  
    df['cos_hour'] = np.cos(2*np.pi *(df['date']).apply(hr_func)/24)
    df['sin_hour'] = np.sin(2*np.pi *(df['date']).apply(hr_func)/24)
    
    df['cos_day'] = np.cos(2*np.pi *(df['date']).apply(day_func)/365)
    df['sin_day'] = np.sin(2*np.pi *(df['date']).apply(day_func)/365)    
    
    df['cos_month'] = np.cos(2*np.pi *df['date'].apply(month_func)/12)
    df['sin_month'] = np.sin(2*np.pi *df['date'].apply(month_func)/12)
     
    df['cos_wd'] = np.cos(np.pi * df['wd']/180)
    df['sin_wd'] = np.sin(np.pi * df['wd']/180)
    
    df['ws'] = np.sqrt(df['ws'])
#     df['ws3'] = df['ws']**3
    
    df['u2'] = df['u']**2
#     df['u3'] = df['u']**3
    
#     df['v2'] = df['v']**2
#     df['v3'] = df['v']**3

    return df


# def forecast_nb_to_predict(df, target_dates, start_forecastdate):
#     """Returns a list of the forecast number to be predicted for the contest."""
#     first_nb = df[df.forecast_time >= start_forecastdate].head(1).forecast.values[0]
#     nb_forecast = int(len(target_dates)/48)
#     cast_predict = [first_nb]
#     for i in range(1, nb_forecast):
#         cast_predict.append(cast_predict[i-1] + 7)
#     return cast_predict


# def forecast_nb_to_predict_corr(df, test_dates):
#     """Returns a list of the forecast number to be predicted for the contest."""
#     df[df.date.isin(test_dates.date.unique())]
#     return cast_predict



def forecast_batch(df):
    """Creates a batch number feature for each 48h predicted."""
    i = 1
    for date in df.date.unique():
        df.loc[df.date == date, 'forecast'] = i
        i += 1
    return df 

def forecast_distance(df):
    """Computes the distance from the time of forecast and the date forecast. 
    The incertitude of the forecast increases with the distance"""
    df.sort_values(by = ['forecast', 'date'], inplace = True)
    for cast in df.forecast.unique():
        nb_forecast = len(df.loc[df.forecast == cast, 'date'])
        i = 0
        for date in df[df.forecast == cast].date.unique():
            df.loc[(df.forecast == cast)&(df.date == date), 'forecast_dist'] = i
            i+=1  
    return df

def wp_getter(df, target, target_wp_col):
    for date in target.date.unique():
         df.loc[df.date == date, 'wp'] = target.loc[target.date == date, target_wp_col].values[0]
    return df

    
def rolling_window_48h(df):
    """Makes a rolling window for each batch of 48h forecast. 
    The goal here is to take the 36h prior which have the smallest distance to forecast time. This way, 
    the forecast (u,v,ws...) used is more accurate."""
    df_rolled = pd.DataFrame()   
    nb_48h = int(len(df)/48 + 1)
    for i in range(4, nb_48h): 
            condition = (df.forecast >= i-3) & (df.forecast <= i)
            forecast_48 = df[condition].sort_values(by = ['date', 'forecast_time'], ascending=[True, False])
            forecast_48 = forecast_48.drop_duplicates(subset = 'date')
            forecast_48 = rolling_windows(forecast_48)
            forecast_48 = forecast_48[forecast_48.forecast == i]
            df_rolled = pd.concat([df_rolled, forecast_48], ignore_index=True)
    return df_rolled



def rolling_windows(df):
    # Wind speed
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 36]:
        df['ws_T_' + str(i)] = df['ws'].shift(i)     
    
    for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 36]:
        df['ws_T_' + str(i) + '_mean'] = df['ws'].rolling(window = i).mean() 
        
    for i in [2, 3, 4, 5, 6, 12, 24, 36]:
        df['ws_T_' + str(i) + '_std'] = df['ws'].rolling(window = i).std()
        
    for i in [2, 3, 4, 5, 6, 12, 24, 36]:
        df['ws_T_' + str(i) + '_median'] = df['ws'].rolling(window = i).median()
        
    for i in [2, 3, 4, 5, 6, 12, 24, 36]:
        df['ws_T_' + str(i) + '_max'] = df['ws'].rolling(window = i).max()    
        
    for i in [2, 3, 4, 5, 6, 12, 24, 36]:
        df['ws_T_' + str(i) + '_min'] = df['ws'].rolling(window = i).min()         
       
    # u 
    for i in [1, 2, 3, 4, 5, 6, 12, 24, 36]:
        df['coswd_' + str(i)] = df['cos_wd'].shift(i)
        
    for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 36]:
        df['coswd_' + str(i) + '_mean'] = df['cos_wd'].rolling(window = i).mean()  
        
    for i in [2, 3, 4, 5, 6, 12, 24, 36]:
        df['coswd_' + str(i) + '_std'] = df['cos_wd'].rolling(window = i).std()
        
    for i in [2, 3, 4, 5, 6, 12, 24, 36]:
        df['coswd_' + str(i) + '_median'] = df['cos_wd'].rolling(window = i).median() 
        
    for i in [2, 3, 4, 5, 6, 12, 24, 36]:
        df['coswd_' + str(i) + '_max'] = df['cos_wd'].rolling(window = i).max()    
        
    for i in [2, 3, 4, 5, 6, 12, 24, 36]:
        df['coswd_' + str(i) + '_min'] = df['cos_wd'].rolling(window = i).min()      
    
    
    # u 
    for i in [1, 2, 3, 4, 5, 6, 12, 24, 36]:
        df['u_T_' + str(i)] = df['u'].shift(i)
        
    for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 36]:
        df['u_T_' + str(i) + '_mean'] = df['u'].rolling(window = i).mean()  
        
    for i in [2, 3, 4, 5, 6, 12, 24, 36]:
        df['u_T_' + str(i) + '_std'] = df['u'].rolling(window = i).std()
        
    for i in [2, 3, 4, 5, 6, 12, 24, 36]:
        df['u_T_' + str(i) + '_median'] = df['u'].rolling(window = i).median() 
        
    for i in [2, 3, 4, 5, 6, 12, 24, 36]:
        df['u_T_' + str(i) + '_max'] = df['u'].rolling(window = i).max()    
        
    for i in [2, 3, 4, 5, 6, 12, 24, 36]:
        df['u_T_' + str(i) + '_min'] = df['u'].rolling(window = i).min()     
    
    
        # u 
    for i in [1, 2, 3, 4, 5, 6, 12, 24, 36]:
        df['u2_T_' + str(i)] = df['u2'].shift(i)
        
    for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 36]:
        df['u2_T_' + str(i) + '_mean'] = df['u2'].rolling(window = i).mean()  
        
    for i in [2, 3, 4, 5, 6, 12, 24, 36]:
        df['u2_T_' + str(i) + '_std'] = df['u2'].rolling(window = i).std()
        
    for i in [2, 3, 4, 5, 6, 12, 24, 36]:
        df['u2_T_' + str(i) + '_median'] = df['u2'].rolling(window = i).median() 
        
    for i in [2, 3, 4, 5, 6, 12, 24, 36]:
        df['u2_T_' + str(i) + '_max'] = df['u2'].rolling(window = i).max()    
        
    for i in [2, 3, 4, 5, 6, 12, 24, 36]:
        df['u2_T_' + str(i) + '_min'] = df['u2'].rolling(window = i).min()    
    
    
    # v
    for i in [1, 2, 3, 4, 5, 6, 12, 24, 36]:
        df['v_T_' + str(i)] = df['v'].shift(i)
        
    for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 36]:
        df['v_T_' + str(i) + '_mean'] = df['v'].rolling(window = i).mean()  
        
    for i in [2, 3, 4, 5, 6, 12, 24, 36]:
        df['v_T_' + str(i) + '_std'] = df['v'].rolling(window = i).std()   
    
    for i in [2, 3, 4, 5, 6, 12, 24, 36]:
        df['v_T_' + str(i) + '_median'] = df['v'].rolling(window = i).median() 
    
    for i in [2, 3, 4, 5, 6, 12, 24, 36]:
        df['v_T_' + str(i) + '_max'] = df['v'].rolling(window = i).max()    
        
    for i in [2, 3, 4, 5, 6, 12, 24, 36]:
        df['v_T_' + str(i) + '_min'] = df['v'].rolling(window = i).min()  

    return df


class FeaturesPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, target, train_end_date, test_start_date):
        self.target = target
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date
        
    def fit(self):
        return self
        
    def transform(self, features, target_col):
        df = forecast_batch(features)
        df = date_conversion(df)
        df = forecast_distance(df)
        print(f'---------------Attribute addition-----------\n')
        df = feature_engineering(df)
        df = wp_getter(df, self.target, target_col)
        print(f'---------------Rolling Window---------------\n')
        df = rolling_window_48h(df)
        print(f'------------Train/Test Separation-----------\n')
        train = df[(df.forecast_time < self.train_end_date)]
        train = train[~((train.wp <=0) & (train.ws > 3.3))]
        test = df[(df.date >= self.test_start_date)]
        return train, test