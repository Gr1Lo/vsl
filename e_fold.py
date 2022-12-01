import numpy as np
from numba import njit
import gc

@njit(parallel=True)
def e_fold(grid, sort_lon, sort_lat):
  if 1:
    cou = 0
    done_list = []
    corr_list_1 = []
    dist_list_1 = []
    for i_lat in range(0,len(sort_lat)):
      for i_lon in range(0,len(sort_lon)):
        done_list.append([i_lat,i_lon])
        m_g = grid[:,i_lat,i_lon]
        if ~np.isnan(m_g[0]):
          lat_coor = sort_lat[i_lat]
          lon_coor = sort_lon[i_lon]
          for i_lat0 in range(0,len(sort_lat)):
              for i_lon0 in range(0,len(sort_lon)):
                m_g0 = grid[:,i_lat0,i_lon0]
                if (~np.isnan(m_g0[0])) and ([i_lat0,i_lon0] not in done_list):
                  lat_coor0 = sort_lat[i_lat0]
                  lon_coor0 = sort_lon[i_lon0]

                  dist = (((lat_coor-lat_coor0)**2)**0.5 + ((lon_coor-lon_coor0)**2)**0.5)**0.5

                  if dist <= 3000000:
                    corr_p = np.round(np.corrcoef(m_g0, m_g)[0][1],3)
                    if np.abs(corr_p)>0.2:
                      corr_list_1.append(np.abs(corr_p))
                      dist_list_1.append(dist)

        print(i_lat)
        cou += 1

  return dist_list_1, corr_list_1



import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import statsmodels.api as sm
from matplotlib.pyplot import figure

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def plot_e(dist_list, corr_list):
  figure(figsize=(9, 5), dpi=200)
  lowess = sm.nonparametric.lowess(corr_list, dist_list, frac=.3)

  # unpack the lowess smoothed points to their values
  lowess_x = list(zip(*lowess))[0]
  lowess_y = list(zip(*lowess))[1]
  v, idx = find_nearest(lowess_y, 1/np.exp(1))

  plt.scatter(dist_list, corr_list, s=1, color='red')
  plt.plot(lowess_x, lowess_y)
  plt.plot(lowess_x[idx], v, 'g*')
  plt.text(lowess_x[idx], v+0.005, '({}, {})'.format(np.round(lowess_x[idx],3), 
                                                     np.round(v,3)))
  plt.show()
