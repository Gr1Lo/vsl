from wpca import PCA, WPCA, EMPCA
from sklearn.linear_model import LinearRegression, HuberRegressor
import numpy as np
import pandas as pd

def max_min_year_trsgi(trsgi_df):
  maxv = 0
  minv = 99999
  for index, row in trsgi_df.iterrows():
      cur_years = row['ages'].values
      if max(cur_years) > maxv:
        maxv = max(cur_years)

      if min(cur_years) < minv:
        minv = min(cur_years)
  
  return minv,maxv


def eppr(grid, trsgi_df, sort_lon, sort_lat, search_rad, mask, single = False):
  #single - select the most correlated pseuproxy
  result = np.empty(shape=(len(m_mask[~m_mask]),grid.shape[1],grid.shape[2]))
  result[:] = np.nan

  p_list = [0,0.1,0.25,0.5,0.67,1,1.5,2]
  
  trsgi_df1 = trsgi_df.copy()
  st_year, end_year = max_min_year_trsgi(trsgi_df)
  range_years = list(range(st_year,end_year+1))

  #filling missing years by zeros
  for index, row in trsgi_df1.iterrows():
      cur_years = row['ages'].values
      temp3 = []
      trsgi_n = []
      corr_n = []
      for i in range(len(range_years)):
          if range_years[i] not in cur_years:
              temp3.append(0)
              trsgi_n.append(0)
              corr_n.append(0)
          else:
              temp3.append(range_years[i])
              trsgi_n.append(row['trsgi'].values[i])
              corr_n.append(1)

      row['ages'] = temp3
      row['trsgi'] = trsgi_n
      row['corr'] = corr_n

  cou = 0
  for i_lat in range(0,len(sort_lat)):
    for i_lon in range(0,len(sort_lon)):
      cou+= 1
      if 1:
        print(cou)
        m_g = grid[:,i_lat,i_lon][m_mask]
        m_g0 = grid[:,i_lat,i_lon]
        if all(~np.isnan(v) for v in m_g):
          lat_coor = sort_lat[i_lat]
          lon_coor = sort_lon[i_lon]
          trsgi_df1['r'] = trsgi_df1.apply(lambda row: ((row.geo_meanLat-lat_coor)**2 + 
                                          (row.geo_meanLon-lon_coor)**2)**0.5, axis=1)
          if single:
            trsgi_df1['corr'] = trsgi_df1.apply(lambda row: np.round(np.corrcoef
                                                                      (np.array(row['trsgi'])[m_mask],m_g)
                                                                      [0][1],3), axis=1)
            #c_maxes = trsgi_df1.groupby(['geo_meanLat', 'geo_meanLon'])['corr'].transform(max)
            trsgi_df1 = trsgi_df1.sort_values('corr').drop_duplicates(
                subset=['geo_meanLat', 'geo_meanLon'], keep='last')


          trsgi_df2 = trsgi_df1[trsgi_df1['r']<=search_rad]
          if len(trsgi_df2) < 10:
            trsgi_df2 = trsgi_df1.sort_values('r',ascending = True).head(10)

          ############
          preds = []
          for p in p_list:
            uTR_list = []
            corr_list = []
            corr_p_l = []
            for index, row in trsgi_df2.iterrows():
                uTR_list.append(row['trsgi'])
                corr_p = np.round(np.corrcoef(np.array(row['trsgi'])[m_mask], m_g)[0][1],3)
                corr_p_l.append([corr_p+0.0011]*len(row['trsgi']))
                #corr_list.append(np.array(row['corr'])*(np.abs(corr_p)+0.001))

            corr_list = np.abs(corr_p_l)
            uTR_arr = np.array(uTR_list).T
            corr_arr = np.abs(np.array(corr_list).T) ** p 

            wTR_arr = uTR_arr * corr_arr
            t_corr_arr = np.array(corr_list).T

            corr_arr = np.where(t_corr_arr==0, 0, 1)
            pc_sum=0
            n_c = 3
            while pc_sum < 0.9:
              principalComponents = WPCA(n_components=n_c).fit(wTR_arr[m_mask], weights = corr_arr[m_mask])
              pc_sum = np.sum(principalComponents.explained_variance_ratio_)
              n_c+=2
              '''print('res ',pc_sum, n_c)
              if n_c == 11:
                principalComponents = WPCA(n_components=10).fit(wTR_arr[m_mask], weights = corr_arr[m_mask])
                pc_sum = 1'''

            principalComponents0 = principalComponents.transform(wTR_arr)
            principalDf0 = pd.DataFrame(data = principalComponents0)
            zerosDf0 = pd.DataFrame(data = uTR_arr)

            tr_df0 = principalDf0[mask]
            tr_m_g = m_g0[:end_year+1-st_year][mask]
            test_df0 = principalDf0[~mask]
            test_m_g = m_g0[:end_year+1-st_year][~mask]

            tr_zerosDf0 = zerosDf0[mask]
            test_zerosDf0 = zerosDf0[~mask]

            #drop all zeros
            tr_df = tr_df0[~(tr_zerosDf0 == 0).all(axis=1)]
            tr_m_g = tr_m_g[~(tr_zerosDf0 == 0).all(axis=1)]
            test_df = test_df0[~(test_zerosDf0 == 0).all(axis=1)]
            test_m_g = test_m_g[~(test_zerosDf0 == 0).all(axis=1)]
            fill_v = len(test_df0)-len(test_m_g)

            #lm
            reg = LinearRegression().fit(tr_df.to_numpy(), tr_m_g)
            pred = reg.predict(test_df.to_numpy())
            preds.append(pred)

          #print(preds)
          preds = np.mean(preds, axis=0)
          preds = np.append(preds, np.repeat(np.nan, fill_v))
          result[:,i_lat,i_lon] = preds

  return result
