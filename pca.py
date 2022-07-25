import warnings
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt
from pyEOF import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def visualization(da, pcs, eofs_da, evf, n, dims):
    fig = plt.figure(figsize = (12,2.5*n))

    ax = fig.add_subplot(n+1,2,1)
    da.mean(dim=dims).plot(ax=ax)
    ax.set_title("average scpdsi")

    ax = fig.add_subplot(n+1,2,2)
    da.mean(dim="time").plot(ax=ax)
    ax.set_title("average scpdsi")

    for i in range(1,n+1):
        pc_i = pcs["PC"+str(i)].to_xarray()
        eof_i = eofs_da.sel(EOF=i)["scpdsi"]
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
          plots=False):

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
        visualization(ds_n, pcs, eofs_da, evfs, n, dims)

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
