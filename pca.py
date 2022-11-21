import warnings
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt
from pyEOF import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy


def visualization(da, pcs, eofs_da, evf, n, dims, var_name = 'scpdsi'):
    fig = plt.figure(figsize = (12,2.5*n))

    ax = fig.add_subplot(n+1,2,1)
    da.mean(dim=dims).plot(ax=ax)
    ax.set_title("average " + var_name)

    ax = fig.add_subplot(n+1,2,2)
    da.mean(dim="time").plot(ax=ax)
    ax.set_title("average " + var_name)

    for i in range(1,n+1):
        pc_i = pcs["PC"+str(i)].to_xarray()
        eof_i = eofs_da.sel(EOF=i)[var_name]
        frac = str(np.array(evf[i-1]*100).round(2))

        ax = fig.add_subplot(n+1,2,i*2+1)
        pc_i.plot(ax=ax)
        ax.set_title("PC"+str(i)+" ("+frac+"%)")

        ax = fig.add_subplot(n+1,2,i*2+2)
        eof_i.plot(ax=ax,
                   vmin=-0.75, vmax=0.75, cmap="RdBu_r",
                   cbar_kwargs={'label': ""})
        ax.set_title("EOF"+str(i)+" ("+frac+"%)")

    plt.tight_layout()
    plt.show()

def eof_an(df_clim_index, ds_n, n = 10, scale_type = 0, pca_type = "varimax", evfs_lower_limit = 0, dims = ["latitude","longitude"],
          plots=False, var_name = 'scpdsi'):

    '''
    EOF-анализ
    df_clim_index - переменная со значениями климатических индексов из r_netCDF()
    n - минимальное количество EOF
    scale_type - параметр, отвечающий за масштабирование при расчете EOF
    evfs_lower_limit - нижний порог объясненной доли дисперсии
    dims - названия осей с координатами
    plots - выводить изображения EOF
    '''
    evfs = [0]

    while np.sum(evfs)<=evfs_lower_limit:
        pca = df_eof(df_clim_index,pca_type=pca_type,n_components=n)
        evfs = pca.evf(n=n)
        print('Количество EOF: '+ str(n), '; Доля объясненной дисперсии: ' + str(round(np.sum(evfs),3)))
        n = n+1
    
    n = n-1

    eofs = pca.eofs(s=scale_type, n=n)

    eofs_da = eofs.stack(dims).to_xarray()
    pcs = pca.pcs(s=scale_type, n=n)
    eigvals = pca.eigvals(n=n)

    # plot
    if plots:
        visualization(ds_n, pcs, eofs_da, evfs, n, dims, var_name)

    return (pca, eofs, pcs, evfs, eigvals, n)
  
  
def pca_tr(trsgi, n = 10, pcas_lower_limit = 0):
    '''
    PCA-анализ
    trsgi - значения индекса trsgi
    n - минимальное количество PCA
    pcas_lower_limit - нижний порог доли объясненной дисперсии
    '''

    x = StandardScaler().fit_transform(trsgi)
    pcas = [0]
    while np.sum(pcas)<=pcas_lower_limit:
        pca = PCA(n_components=n)
        principalComponents = pca.fit_transform(x)
        print('Количество компонент: '+ str(n), '; Доля объясненной дисперсии: ' + str(round(np.sum(pca.explained_variance_ratio_),3)))
        pcas = pca.explained_variance_ratio_
        n = n+1
    
    n = n-1

    pca = PCA(n_components=n)
    principalComponents = pca.fit_transform(x)

    return principalComponents,pca.explained_variance_ratio_

def rot_check(y_pred, y_true, eofs, eigvals, pca, ds_n, ttl, p_type = 'diff', scale_type = 2):
       
        '''
        Визульная оценка работы моделей
        y_pred - значения главных компонент, 
        y_true - реальные значения, 
        eofs - набор значений функций EOF, 
        eigvals - собственные числа EOF, 
        pca - объект, полученный в результате eof_an, 
        ds_n -трехмерный массив, используется для извлечения параметров исходных данных, 
        ttl - название, 
        p_type - параметр, на основе которого будут строиться карты
                  'diff' - модуль разницы
                  'corr' - коэффициент корреляции Пирсона
        scale_type - параметр отвечающий за масштабирование главных компонент и 
                     EOF через умножение/деление значений на собственные числа
        '''

        if scale_type == 2:
          eofs = eofs[0:len(eofs)] / np.sqrt(eigvals[0:len(eofs)])[:, np.newaxis]
          pcs = y_pred[:, 0:len(eofs)] / np.sqrt(eigvals[0:len(eofs)])

        if scale_type == 1:
          eofs = eofs[0:len(eofs)] * np.sqrt(eigvals[0:len(eofs)])[:, np.newaxis]
          pcs = y_pred[:, 0:len(eofs)] * np.sqrt(eigvals[0:len(eofs)])

        Yhat = np.dot(pcs, eofs.to_numpy())
        Yhat = pca._scaler.inverse_transform(Yhat)
        u = Yhat
        u0 = y_true

        if p_type=='corr':
          coor_ar = []
          for i in range(u0.shape[1]):
            i0 = u[:,i]
            i1 = u0[:,i]
            
            if ~np.isnan(i0[0]):
              corr2 = scipy.stats.pearsonr(i0,i1)[0]
              coor_ar.append(corr2)
            else:
              coor_ar.append(np.nan)

          loss0 = np.array(coor_ar)
          ttl_str = '; среднее значение коэффициента корреляции = '
          vmin = -1
          vmax = 1

        else:
          loss0 = np.abs(u - u0)
          loss0 = np.where(loss0>50, np.nan, loss0)
          loss0 = np.mean(loss0,axis=0)
          ttl_str = '; среднее значение разницы = '
          vmin = 0
          vmax = 8

        
        new = np.reshape(loss0, (-1, ds_n.shape[2]))
        plt.figure(figsize = (19,10))
        im = plt.imshow(new[::-1], interpolation='none',
                        vmin=vmin, vmax=vmax,cmap='jet')

        cbar = plt.colorbar(im,
                            orientation='vertical')
        plt.axis('off')
        plt.tight_layout()

        loss0 = np.nanmean(loss0)

        plt.title(ttl + ttl_str + str(round(loss0,3)),fontsize=20)
        plt.show()

        return loss0
