import keras
from keras import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
import tensorflow as tf
from sklearn.utils import shuffle
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from matplotlib import colors
import numpy as np
from matplotlib import pyplot as plt
import scipy

def train_and_test(trsgi, labels, p_v, keep_order = False):

  '''
  Разделение выборки на тренировочную и тестовую
  trsgi - значения предикторов,
  labels - предсказываемые значения,
  p_v - доля тестовой выборки,
  keep_order - использовать для теста первые записи (сохранять порядок) 
  '''
  
  np.random.seed(123)

  nums = np.ones(len(trsgi))
  nums[:int(len(trsgi)*p_v)] = 0
  if ~keep_order:
    np.random.shuffle(nums)

  mask = 1 == nums
  mask = np.array(mask)

  train_trsgi = trsgi[mask]
  train_labels = labels[mask]
  test_trsgi = trsgi[~mask]
  test_labels = labels[~mask]

  return train_trsgi, train_labels, test_trsgi, test_labels

#функция формирования тренировочного набора для RNN
def make_x_y(ts, data):
    """
    Parameters
    ts : число шагов для RNN
    data : numpy array с предикторами

    x - наборы предикторов, в том числе и с предыдущих шагов
    y - предикторы только с действующего шага
    """
    x, y = [], []
    offset = 0
    for i in data:
        if offset < len(data)-ts:
            x.append(data[offset:ts+offset])
            y.append(data[ts+offset])
            offset += 1

    return np.array(x), np.array(y)
  
  
def m_loss_func_weight(l_evfs, use_w):
    '''
    Кастомная функция потерь
    '''
    def loss(y_true, y_pred):

        evfs = l_evfs
        if use_w==False:
          evfs = np.ones(len(evfs))

        Yhat = y_pred * evfs
        Yhat0 = y_true * evfs

        loss0 = tf.math.square(tf.math.subtract(Yhat,Yhat0))
        loss0 = tf.math.reduce_mean(loss0,axis=1) 

        return loss0

    return loss

def simp_net_regression_1(trsgi_values, resp, ttl, model, shuffle_b, evfs = None, use_w=False, min_delta = 0.1):

    '''
    Запуск обучения модели регрессии
    trsgi_values - набор значений, полученных по ДКХ
    resp - набор предсказываемых значений
    ttl - название графика, описывающего ход обучения
    model - модель, сформированная через get_model_*
    shuffle_b - перемешивание очередности датасета
    evfs - доля объясненной диспесии каждой EOF
    use_w - использовать evfs в качестве весов перед расчетом функции потерь
    min_delta - пороговое значение изменение функции потерь при обучении
    '''

    np.random.seed(123)
    tf.random.set_seed(1234)

    trsgi = trsgi_values[:]
    all_arr = resp[:]

    trsgi_values = np.asarray(trsgi_values)
    all_arr = np.asarray(all_arr)
    
    model.compile(optimizer='adam',
                  run_eagerly=True,
                  loss = m_loss_func_weight(l_evfs=evfs,use_w=use_w))

    if shuffle_b:
      trsgi_values, all_arr = shuffle(trsgi_values, all_arr)
      callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=min_delta)
      v_s = 0.2
    
    else:
      trsgi_values, y = make_x_y(1, trsgi_values)
      callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=min_delta)
      v_s = 0.2
      
    history = model.fit(trsgi_values,
                            all_arr,
                            verbose=0,
                            epochs=100, batch_size = 10,
                            shuffle=shuffle_b,
                            validation_split = v_s,
                            callbacks=[callback]
                            )

    return model, history


def get_model_regression_1(n_inputs, n_outputs, use_drop = False, use_batch_norm = False):

  '''
  Описание сети для задачи регрессии
  n_inputs - число предикторов,
  n_outputs - количество предсказываемых значений, по умоляанию 1
  use_drop - параметр, отвечающий за рандомное отключение доли нейронов (30%)
  use_batch_norm - параметр, отвечающий за использование batch-нормализации
  '''

  model = Sequential()

  model.add(Dense(300, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
  if use_batch_norm == True:
    model.add(BatchNormalization())
  if use_drop == True:
    model.add(Dropout(0.30))

  model.add(Dense(110, kernel_initializer='he_uniform', activation='relu'))
  if use_batch_norm == True:
    model.add(BatchNormalization())
  if use_drop == True:
    model.add(Dropout(0.30))

  model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
  return model

def get_model_regression_RNN(n_inputs, n_outputs = 1, use_drop = False, use_batch_norm = False,
                             inp_shp=None):

  '''
  Описание рекурсивной сети для задачи регрессии
  n_inputs - число предикторов,
  n_outputs - количество предсказываемых значений, по умоляанию 1
  use_drop - параметр, отвечающий за рандомное отключение доли нейронов (30%)
  use_batch_norm - параметр, отвечающий за использование batch-нормализации,
  inp_shp - размаер входных данных
  '''


  model = Sequential()
  lstm = keras.layers.LSTM(100, input_shape=(inp_shp[1], inp_shp[2]), recurrent_dropout=0.1, 
                           return_sequences=True)

  model.add(lstm)
  if use_batch_norm == True:
    model.add(BatchNormalization())
  if use_drop == True:
    model.add(Dropout(0.20))

  #model.add(keras.layers.RepeatVector(inp_shp[1]))
  # decoder layer
  model.add(keras.layers.LSTM(100, activation='relu', return_sequences=True))
  model.add(keras.layers.TimeDistributed(Dense(n_outputs)))

  return model


def run_model(pcs_CMIP6_1, vsl_1000_pc, evfs_CMIP6_1, type_m, use_w=False):
      '''
      Запуск обучения моделей
      pcs_CMIP6_1 - значения главных компонент scpdsi
      vsl_1000_pc - значения главных компонент trsgi
      evfs_CMIP6_1 - массив с долей объясненных дисперсий каждой компонентой,
      type_m - вариант модели:
          'NN' - стнадартная нейронная сеть
          'RNN' - реккурентная нейронная сеть
          'lm' - линейниая модель
      use_w - расчитывать функцию потерь с учетом evfs_CMIP6_1

      Возвращает:
      inverse_te_l - тестовые значения главных компонент scpdsi
      inverse_est - модельная оценка главных компонент scpdsi
      '''

      # create scaler
      scaler = MinMaxScaler()
      scaler_t = MinMaxScaler()
      scaler_vsl_1000 = MinMaxScaler()
      scaler_t_vsl_1000 = MinMaxScaler()

      # fit and transform in one step

      normalized_vsl_1000 = scaler_vsl_1000.fit_transform(pcs_CMIP6_1.to_numpy())
      vsl_1000_pc_normalized = scaler_t_vsl_1000.fit_transform(vsl_1000_pc)

      target = normalized_vsl_1000
      n_inputs = vsl_1000_pc_normalized.shape[1]

      if type_m=='NN':
        model = get_model_regression_1(n_inputs,
                                      n_outputs = pcs_CMIP6_1.shape[1],
                                      use_drop = True,
                                      use_batch_norm = True)

        tr_t, tr_l, te_t, te_l = train_and_test(vsl_1000_pc_normalized, target, 0.2)

        model, history = simp_net_regression_1(tr_t, tr_l, 
                                              'test', model, 
                                              shuffle_b=True,
                                              evfs = evfs_CMIP6_1, 
                                              use_w=use_w)
        score = model.evaluate(te_t, te_l, verbose=0)

        inverse_te_l = scaler_vsl_1000.inverse_transform(te_l)
        est = model.predict(te_t)
        inverse_est = scaler_vsl_1000.inverse_transform(est)

      elif type_m=='RNN':
        trsgi_values, y = make_x_y(1, vsl_1000_pc_normalized)

        inp_shp = trsgi_values.shape
        model = get_model_regression_RNN(n_inputs,
                                        n_outputs = target.shape[1],
                                        use_drop = True,
                                        use_batch_norm = True,
                                        inp_shp=inp_shp)
                    
        tr_t, tr_l, te_t, te_l = train_and_test(vsl_1000_pc_normalized, normalized_vsl_1000, 0.2, keep_order=True)

        model, hystory = simp_net_regression_1(tr_t, tr_l, 
                                              'test', model, shuffle_b = False,
                                              use_w=use_w,
                                              evfs = evfs_CMIP6_1, min_delta = 0.0001)
        
        te_t, y = make_x_y(1, te_t)
        inverse_te_l = scaler_vsl_1000.inverse_transform(te_l)
        est = model.predict(te_t)
        inverse_est = scaler_vsl_1000.inverse_transform(est[:,0,:])

      elif type_m=='lm':
        tr_t, tr_l, te_t, te_l = train_and_test(vsl_1000_pc_normalized, target, 0.2)
        model_0 = sm.OLS(tr_l,tr_t)

        results = model_0.fit()
        inverse_te_l = scaler_vsl_1000.inverse_transform(te_l)
        est = results.predict(te_t)
        inverse_est = scaler_vsl_1000.inverse_transform(est)

      return inverse_te_l, inverse_est
    
    
def rev_diff(y_pred, y_true, eofs, eigvals, pca, ds_n, ttl, p_type='diff', scale_type = 2):
       
        '''
        Визульная оценка работы моделей
        y_pred - тестовые значения главных компонент scpdsi, 
        y_true - модельная оценка главных компонент scpdsi, 
        eofs - набор значений функций EOF, 
        eigvals - собственные числа EOF, 
        pca - объект, полученный в результате eof_an, 
        ds_n -трехмерный массив, используется для извлечения параметров исходных данных, 
        ttl - название модели, 
        p_type - параметр, на основе которого будут строиться карты
                  'diff' - модуль разницы
                  'corr' - коэффициент корреляции Пирсона
        scale_type - параметр отвечающий за масштабирование главных компонент и 
                     EOF через умножение/деление значений на собственные числа
        '''

        if scale_type == 2:
          eofs = eofs[0:len(eofs)] / np.sqrt(eigvals[0:len(eofs)])[:, np.newaxis]
          pcs = y_pred[:, 0:len(eofs)] / np.sqrt(eigvals[0:len(eofs)])
          pcs0= y_true[:, 0:len(eofs)] / np.sqrt(eigvals[0:len(eofs)])

        if scale_type == 1:
          eofs = eofs[0:len(eofs)] * np.sqrt(eigvals[0:len(eofs)])[:, np.newaxis]
          pcs = y_pred[:, 0:len(eofs)] * np.sqrt(eigvals[0:len(eofs)])
          pcs0 = y_true[:, 0:len(eofs)] * np.sqrt(eigvals[0:len(eofs)])

        Yhat = np.dot(pcs, eofs.to_numpy())
        Yhat = pca._scaler.inverse_transform(Yhat)
        u = Yhat

        Yhat0 = np.dot(pcs0, eofs.to_numpy())
        Yhat0 = pca._scaler.inverse_transform(Yhat0)
        u0 = Yhat0

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
    
    
