import os
import pandas as pd
import pickle
import xarray as xr
from pyEOF import *
from sklearn import linear_model

def r_execel(f_path, drop_val = 2):
    '''
    Формирование двумерного массива по годам из данных,
    полученных из ДКХ с заполнением пустых значений линейной регрессией
    f_path - путь до xlsx файла
    drop_val - число строчек, которые будут удалены с конца табилцы
    '''
    df = pd.read_excel(f_path, index_col=None)
    fn_list = df['file_name'].unique()

    trsgi_values = []
    for i in (range(1901, np.max(df['age'])+1)):
      one_year_arr = []
      print(i)
      df0 = df[df['age'] == i]
      for j in fn_list:
        re = df0[df0['file_name'] == j]['trsgi'].values
        if len(re)>0:
          one_year_arr.append(re[0])
        else:
          one_year_arr.append(None)

      trsgi_values.append(one_year_arr)

    df_trsgi_values = pd.DataFrame(data=trsgi_values)
    ind_list = []
    for index, val in df_trsgi_values.isna().sum().iteritems():
      if val < drop_val+1:
        ind_list.append(index)

    df_trsgi_values.drop(df_trsgi_values.tail(drop_val).index,inplace=True)
    arrX = df_trsgi_values[ind_list].to_numpy()

    m_list = []
    for i in range(len(df_trsgi_values.columns)):
      arrY = df_trsgi_values[i].to_numpy()
      ind_NONE = np.where(np.isnan(arrY))
      ind_not_NONE = np.where(~np.isnan(arrY))

      regr = linear_model.LinearRegression()
      regr.fit(arrX[ind_not_NONE], arrY[ind_not_NONE])
      if len(ind_NONE[0])>0:
        arrY[ind_NONE] = np.around(regr.predict(arrX[ind_NONE]),3)
      m_list.append(arrY)

    mat = np.array(m_list)
    res = np.transpose(mat)

    return res


def r_netCDF(f_path, min_lon = -145, min_lat = 14, max_lon = -52, max_lat = 71, swap = 0, force0toNan = False):
    '''
    Формирование таблицы по годам из netCDF с индексами scpdsi
    '''

    ds0 = xr.open_dataset(f_path)
    ds = ds0["scpdsi"]
    n_time = ds0["time"]

    coor = [] 
    for key in ds.coords.keys():
      if key in ('lon','longitude'):
        coor.append(key)
    for key in ds.coords.keys():
      if key in ('lat','latitude'):
        coor.append(key)
    
    #Выбор территории анализа
    mask_lon = (ds.coords[coor[0]] >= min_lon) & (ds.coords[coor[0]] <= max_lon)
    mask_lat = (ds.coords[coor[1]] >= min_lat) & (ds.coords[coor[1]] <= max_lat)

    ret_lon = ds.coords[coor[0]][mask_lon]
    ret_lat = ds.coords[coor[1]][mask_lat]

    ds_n = ds.where(mask_lon & mask_lat, drop=True)
    df_nn = ds_n.to_dataframe().reset_index()
    if force0toNan:
      df_nn[df_nn==0] = np.nan

    df_nn['month'] = np.repeat(n_time.dt.month, len(df_nn)/len(n_time.dt.month))
    df_nn['year'] = np.repeat(n_time.dt.year, len(df_nn)/len(n_time.dt.year))

    #Используется информация только по летним месяцам

    df_nn0 = df_nn[(df_nn['month'] < 9)&(df_nn['month'] > 5)]
    grouped_df = df_nn0.groupby([coor[1], coor[0] ,df_nn0['year']])
    mean_df = grouped_df.mean()
    mean_df = mean_df.reset_index()

    if swap == 0:
      mean_df = mean_df[['year', coor[1], coor[0], 'scpdsi']]
      df_data = get_time_space(mean_df, time_dim = "year", lumped_space_dims = [coor[1],coor[0]])
    else:
      mean_df = mean_df[['year', coor[0], coor[1], 'scpdsi']]
      df_data = get_time_space(mean_df, time_dim = "year", lumped_space_dims = [coor[0],coor[1]])

    return df_data, ds_n, ret_lon, ret_lat

def save_pickle(f_path, vari):
    with open(f_path, 'wb') as f:
        pickle.dump(vari, f)

def read_pickle(f_path):
    with open(f_path, 'rb') as f:
        df_test = pickle.load(f)
        return df_test
