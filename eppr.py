from wpca import PCA, WPCA, EMPCA
from sklearn.linear_model import LinearRegression, HuberRegressor
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.cross_decomposition import PLSRegression


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
  result = np.empty(shape=(len(mask[~mask]),grid.shape[1],grid.shape[2]))
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
        m_g = grid[:,i_lat,i_lon][mask]
        m_g0 = grid[:,i_lat,i_lon]
        if all(~np.isnan(v) for v in m_g):
          lat_coor = sort_lat[i_lat]
          lon_coor = sort_lon[i_lon]
          trsgi_df1['r'] = trsgi_df1.apply(lambda row: ((row.geo_meanLat-lat_coor)**2 + 
                                          (row.geo_meanLon-lon_coor)**2)**0.5, axis=1)
          if single:
            trsgi_df1['corr'] = trsgi_df1.apply(lambda row: np.round(np.corrcoef
                                                                      (np.array(row['trsgi'])[mask],m_g)
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
                corr_p = np.round(np.corrcoef(np.array(row['trsgi'])[mask], m_g)[0][1],3)
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
              principalComponents = WPCA(n_components=n_c).fit(wTR_arr[mask], weights = corr_arr[mask])
              pc_sum = np.sum(principalComponents.explained_variance_ratio_)
              n_c+=2
              '''print('res ',pc_sum, n_c)
              if n_c == 11:
                principalComponents = WPCA(n_components=10).fit(wTR_arr[mask], weights = corr_arr[mask])
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

def closest_node(node, nodes):
    all_p = []
    dist = distance.cdist([node], nodes).min()
    all_p.append(dist)
    return min(all_p)


def eppr_regions(pcs, eofs_da, pcs_test, eofs_da_test, trsgi_df, sort_lon, sort_lat, var_name, mask, thr=0.3):
  all_preds = []
  cou = 0
  for column in pcs:
    print(cou)
    df_pcs = pcs[column]
    df_pcs_test = pcs_test[column]
    
    p_list = [0,0.1,0.25,0.5,0.67,1,1.5,2]
    #p_list = [1]
    
    trsgi_df1 = trsgi_df.copy()
    '''st_year, end_year = max_min_year_trsgi(trsgi_df)
    range_years = list(range(st_year,end_year+1))'''

    #filling missing years by zeros
    trsgi_df1["corr"] = 1
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

    #print(trsgi_df1)
    m_g = df_pcs.values
    m_g_test = df_pcs_test.values
    indices = np.argwhere(np.abs(eofs_da.sel(EOF=cou+1)[var_name].values)> thr)
    t1 = eofs_da.lat[indices[:,0]]
    t2 = eofs_da.lon[indices[:,1]]
    lats_lons = np.array([t1,t2]).T # пары широт и долгот в eof-регионе поиска

    if len(lats_lons)>0: 
      trsgi_df1['r'] = trsgi_df1.apply(lambda row: closest_node([row.geo_meanLat, row.geo_meanLon], lats_lons), 
                                     axis=1)
      trsgi_df2 = trsgi_df1[trsgi_df1['r']<=np.abs(eofs_da.lat[0]-eofs_da.lat[1]).values]
    else:
      m_max = np.nanmax(np.abs(eofs_da.sel(EOF=cou+1)[var_name].values))
      indices = np.argwhere(np.abs(eofs_da.sel(EOF=cou+1)[var_name].values)>= m_max)
      t1 = eofs_da.lat[indices[:,0]]
      t2 = eofs_da.lon[indices[:,1]]
      lats_lons = np.array([t1,t2]).T
      trsgi_df1['r'] = trsgi_df1.apply(lambda row: closest_node([row.geo_meanLat, row.geo_meanLon], lats_lons), 
                                     axis=1)
      trsgi_df2 = trsgi_df1[trsgi_df1['r']<=np.abs(eofs_da.lat[0]-eofs_da.lat[1]).values]
    

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
                #corr_p = np.round(np.corrcoef(row['trsgi'], m_g[:end_year+1-st_year])[0][1],3)
                corr_p = np.round(np.corrcoef(np.array(row['trsgi'])[mask], m_g)[0][1],3)
                corr_p_l.append(corr_p)
                corr_list.append(np.array(row['corr'])*(np.abs(np.repeat(corr_p,len(row['trsgi']))))+0.001)

            uTR_arr = np.array(uTR_list).T
            corr_arr = np.abs(np.array(corr_list).T) ** p 

            wTR_arr = uTR_arr * corr_arr
            t_corr_arr = np.array(corr_list).T
            corr_arr = np.where(t_corr_arr==0, 0, 1)
            pc_sum=0
            n_c = 3

            while pc_sum < 0.9:
              principalComponents = WPCA(n_components=n_c).fit(wTR_arr[mask], 
                                                               weights = corr_arr[mask])
              pc_sum = np.sum(principalComponents.explained_variance_ratio_)
              n_c+=1

            principalComponents0 = principalComponents.transform(wTR_arr)
            principalDf0 = pd.DataFrame(data = principalComponents0)
            zerosDf0 = pd.DataFrame(data = uTR_arr)

            tr_df0 = principalDf0[mask]
            tr_m_g = m_g
            test_df0 = principalDf0[~mask]
            test_m_g = m_g_test

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

    preds = np.mean(preds, axis=0)
    preds = np.append(preds, np.repeat(np.nan, fill_v))
    all_preds.append(preds)
    cou+= 1

  return all_preds


def eppr_PLSR(grid, pcs, eofs_da, trsgi_df, sort_lon, sort_lat, var_name, search_rad, mask, thr=0.3,pick=False):
  result = np.empty(shape=(len(mask[~mask]),grid.shape[1],grid.shape[2]))
  result[:] = np.nan

  all_preds = []
  if 1:
    #p_list = [0,0.1,0.25,0.5,0.67,1,1.5,2]
    p_list = [1]
    
    trsgi_df1 = trsgi_df.copy()
    st_year, end_year = max_min_year_trsgi(trsgi_df)
    range_years = list(range(st_year,end_year+1))

    #filling missing years by zeros
    trsgi_df1["corr"] = 1

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


  #all regions for pixel
  cou1 = 0
  for i_lat in range(0,len(sort_lat)):
    for i_lon in range(0,len(sort_lon)):
        print('n',cou1)
        all_regions = np.array([[1,1]])
        m_g = grid[:,i_lat,i_lon]
        cou1+= 1
        if all(~np.isnan(v) for v in m_g):
          cou = 0
          for column in pcs:
            df_pcs = pcs[column]
            indices = np.argwhere(np.abs(eofs_da.sel(EOF=cou+1)[var_name].values)> thr)
            t1 = eofs_da.lat[indices[:,0]]
            t2 = eofs_da.lon[indices[:,1]]
            lats_lons = np.array([t1,t2]).T # пары широт и долгот в eof-регионе поиска

            if len(lats_lons)>0: 
              closest_in_grid = closest_node([sort_lat[i_lat], sort_lon[i_lon]], lats_lons)
              min_dist = np.abs(eofs_da.lat[0]-eofs_da.lat[1])*1.1
              if min_dist >= closest_in_grid:
                  all_regions = np.append(all_regions, lats_lons, axis=0)

            cou+= 1

          if len(all_regions) > 1:
            all_regions = all_regions[1:]
            new_array = [tuple(row) for row in all_regions]
            regs = np.unique(new_array, axis=0)
            #regs = all_regions
          else:
            regs = [[sort_lat[i_lat], sort_lon[i_lon]]]

          lat_coor = sort_lat[i_lat]
          lon_coor = sort_lon[i_lon]

          trsgi_df1['r'] = trsgi_df1.apply(lambda row: closest_node([row.geo_meanLat, row.geo_meanLon], regs), axis=1)
          trsgi_df2 = trsgi_df1[trsgi_df1['r']<=np.abs(eofs_da.lat[0]-eofs_da.lat[1]).values]

          if len(trsgi_df2) < 10:
            trsgi_df2 = trsgi_df1[trsgi_df1['r']<=search_rad]

          if len(trsgi_df2) < 10:
                  trsgi_df2 = trsgi_df1.sort_values('r',ascending = True).head(10)

          ############
          if 1:
            preds = []
            for p in p_list:
                    uTR_list = []
                    corr_list = []
                    corr_p_l = []
                    for index, row in trsgi_df2.iterrows():
                        uTR_list.append(row['trsgi'])
                        corr_p = np.round(np.corrcoef(np.array(row['trsgi'])[mask], 
                                                      m_g[mask])[0][1],3)
                        corr_p_l.append(corr_p)
                        corr_list.append(np.array(row['corr'])*(np.abs(corr_p)+0.001))

                    uTR_arr = np.array(uTR_list).T
                    corr_arr = np.abs(np.array(corr_list).T) ** p 

                    #wTR_arr = uTR_arr * corr_arr
                    wTR_arr = uTR_arr #* corr_arr
                    t_corr_arr = np.array(corr_list).T
                    corr_arr = np.where(t_corr_arr==0, 0, 1)
                    tr_mask = ~(wTR_arr[mask]==0).all(axis=1)
                    test_mask = ~(wTR_arr[~mask]==0).all(axis=1)

                    
                    #####selecting best_ncomp######
                    best_cor = 0
                    best_ncomp = 0
                    p_v = 0.3
                    nums_tr = np.ones(len(wTR_arr[mask][tr_mask]))
                    nums_tr[:int(len(wTR_arr[mask][tr_mask])*p_v)] = 0
                    np.random.shuffle(nums_tr)
                    mask_tr = 1 == nums_tr
                    m_mask_tr = np.array(mask_tr)
                    if pick:
                      for n_comp in range(1, int(wTR_arr.shape[1]/3), 3):
                        my_plsr = PLSRegression(n_components=n_comp, scale=True)
                        my_plsr.fit(wTR_arr[mask][tr_mask][m_mask_tr], m_g[:end_year+1-st_year][mask][tr_mask][m_mask_tr])
                        preds_PLSR = my_plsr.predict(wTR_arr[mask][tr_mask][~m_mask_tr])
                        cor = np.corrcoef(preds_PLSR.T[0], m_g[:end_year+1-st_year][mask][tr_mask][~m_mask_tr])[0][1]
                        if cor > best_cor:
                          best_cor = cor
                          best_ncomp = n_comp
                    else:
                      best_ncomp = 5

                    best_model = PLSRegression(n_components=best_ncomp, scale=True)
                    best_model.fit(wTR_arr[mask][tr_mask], m_g[:end_year+1-st_year][mask][tr_mask])
                    preds_PLSR = np.empty(shape=len(wTR_arr[~mask]))
                    preds_PLSR[:] = np.nan
                    preds_PLSR[test_mask] = best_model.predict(wTR_arr[~mask][test_mask]).T[0]
                    preds.append(preds_PLSR)

            preds = np.nanmean(preds, axis=0)
            all_preds.append(preds)
            result[:,i_lat,i_lon] = preds

  return result
