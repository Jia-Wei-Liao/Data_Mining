import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import MaxNLocator


def check_special_symbol(df, i, j, symbol_list):
    for s in symbol_list:
        if str(df.iloc[i, j]).rfind(s) != -1:
            return True
    else:
        return False


def get_lower_number(df, i, j, symbol_list):
    n, m = df.shape
    if i < 0:
        return -1, -1
    
    elif j==-1:
        return get_lower_number(df, i-18, m-1, symbol_list)
    
    elif check_special_symbol(df, i, j, symbol_list):
        return get_lower_number(df, i, j-1, symbol_list)
    
    else:
        return i, j

    
def get_hight_number(df, i, j, symbol_list):
    n, m = df.shape
    if i > n:
        return -1, -1
    
    elif j==len(df.columns):
        return get_hight_number(df, i+18, 0, symbol_list)
    
    elif check_special_symbol(df, i, j, symbol_list):
        return get_hight_number(df, i, j+1, symbol_list)
    
    else:
        return i, j

    
def filled_x(df):
    symbol_list = ['#', '*', 'x', 'A']
    n, m = df.shape
    for i in range(n):
        for j in range(m):
            if check_special_symbol(df, i, j, symbol_list):
                lower_index = get_lower_number(df, i, j-1, symbol_list)
                hight_index = get_hight_number(df, i, j+1, symbol_list)
                if lower_index == (-1, -1): lower_index = hight_index
                elif hight_index == (-1, -1): hight_index = lower_index
                df.iloc[i, j] = str((float(df.iloc[lower_index]) + float(df.iloc[hight_index])) / 2)
    
    return df


def get_time_series_label(df, pred_hour_unit, PM2p5=9):
    return np.vstack([df[PM2p5, pred_hour_unit+i] for i in range(df.shape[1]-pred_hour_unit)])


def get_time_series_data(df, cut_unit, pred_hour_unit, PM2p5):
    return np.vstack([df[PM2p5, 0+i:cut_unit+i].reshape(1, -1) for i in range(df.shape[1]-pred_hour_unit)])


def Modeling(data_list, model):
    train_mae_list = []
    test_mae_list = []
    r2_list = []
    
    for tr_X, tr_Y, ts_X, ts_Y in data_list:
        model.fit(tr_X, tr_Y)
        train_pred = model.predict(tr_X)
        test_pred = model.predict(ts_X)
        
        train_mae = metrics.mean_absolute_error(tr_Y, train_pred)
        test_mae = metrics.mean_absolute_error(ts_Y, test_pred)
        r2 = metrics.r2_score(ts_Y, test_pred)
        
        train_mae_list.append(round(train_mae, 4))
        test_mae_list.append(round(test_mae, 4))
        r2_list.append(round(r2, 4))
    
    return train_mae_list, test_mae_list, r2_list


if __name__ == '__main__':
    origin_df = pd.read_csv(open('新竹_2020.csv'))
    origin_df.drop(0, inplace=True)
    origin_df.columns = [e.strip() for e in list(origin_df.columns)]
    origin_df = origin_df.applymap(lambda x: x.strip())

    month = ['10', '11', '12']
    index = origin_df['日期'].map(lambda x: True if x[5:7] in month else False)
    origin_df = origin_df[index]
    df = origin_df.iloc[:, 3:]

    df = filled_x(df).astype(float)
    df[df=='NR'] = 0
    train_index = (31+30)*18
    train_df = df.iloc[:train_index, :]
    test_df = df.iloc[train_index:, :]

    train_X = np.hstack([train_df.iloc[0+18*i:18+18*i, :] for i in range(len(train_df) // 18)])
    test_X = np.hstack([test_df.iloc[0+18*i:18+18*i, :] for i in range(len(test_df) // 18)])
    train_Y_c6_p6 = get_time_series_label(train_X, pred_hour_unit=6)
    test_Y_c6_p6  = get_time_series_label(test_X, pred_hour_unit=6)
    train_Y_c6_p11 = get_time_series_label(train_X, pred_hour_unit=11)
    test_Y_c6_p11  = get_time_series_label(test_X, pred_hour_unit=11)
    train_X_c6_p6_f1 = get_time_series_data(train_X, cut_unit=6, pred_hour_unit=6, PM2p5=9)
    test_X_c6_p6_f1  = get_time_series_data(test_X, cut_unit=6, pred_hour_unit=6, PM2p5=9)
    train_X_c6_p11_f1 = get_time_series_data(train_X, cut_unit=6, pred_hour_unit=11, PM2p5=9)
    test_X_c6_p11_f1  = get_time_series_data(test_X, cut_unit=6, pred_hour_unit=11, PM2p5=9)
    train_X_c6_p6_f18 = get_time_series_data(train_X, cut_unit=6, pred_hour_unit=6, PM2p5=range(18))
    test_X_c6_p6_f18  = get_time_series_data(test_X, cut_unit=6, pred_hour_unit=6, PM2p5=range(18))
    train_X_c6_p11_f18 = get_time_series_data(train_X, cut_unit=6, pred_hour_unit=11, PM2p5=range(18))
    test_X_c6_p11_f18  = get_time_series_data(test_X, cut_unit=6, pred_hour_unit=11, PM2p5=range(18))

    data_list = [
        (train_X_c6_p6_f1, train_Y_c6_p6, test_X_c6_p6_f1, test_Y_c6_p6),
        (train_X_c6_p6_f18, train_Y_c6_p6, test_X_c6_p6_f18, test_Y_c6_p6),
        (train_X_c6_p11_f1, train_Y_c6_p11, test_X_c6_p11_f1, test_Y_c6_p11),
        (train_X_c6_p11_f18, train_Y_c6_p11, test_X_c6_p11_f18, test_Y_c6_p11)
    ]

    lr_model = LinearRegression()
    lr_train_mae, lr_test_mae, lr_r2_list = Modeling(data_list, lr_model)
    xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', verbosity=2)
    xgboost_train_mae, xgboost_test_mae, xgboost_r2_list = Modeling(data_list, xgboost_model)

    print('Training:')
    print(f'Linear Regression MAE: {lr_train_mae}')
    print(f'Xgboost           MAE: {xgboost_train_mae}')
    print('-------------------------------------------')
    print('Testing:')
    print(f'Linear Regression MAE: {lr_test_mae}')
    print(f'Xgboost           MAE: {xgboost_test_mae}')
    print('-------------------------------------------')
    print(f'Linear Regression R-squared: {lr_r2_list}')
    print(f'Xgboost           R-squared: {xgboost_r2_list}')
