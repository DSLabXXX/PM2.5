# _*_ coding: utf-8 _*_

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import time
import keras
# import sys
import os
import cPickle
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate
from keras.layers.core import Dense, Dropout, Flatten
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from reader import read_data_sets, construct_time_steps
from missing_value_processer import missing_check
from feature_processor import data_coordinate_angle
from config import root
root_path = root()


def time_spent_printer(start_time, final_time):
    spent_time = final_time - start_time
    print('totally spent ', end='')
    print(int(spent_time / 3600), 'hours ', end='')
    print(int((int(spent_time) % 3600) / 60), 'minutes ', end='')
    print((int(spent_time) % 3600) % 60, 'seconds')


pollution_site_map = {
    '中部': {'台中': ['大里', '忠明', '沙鹿', '西屯', '豐原'],
           '南投': ['南投', '竹山'],
           '彰化': ['二林', '彰化']},

    '北部': {'台北': ['中山', '古亭', '士林', '松山', '萬華'],
           '新北': ['土城', '新店', '新莊', '板橋', '林口', '汐止', '菜寮', '萬里'],
           '基隆': ['基隆'],
           '桃園': ['大園', '平鎮', '桃園', '龍潭']},

    '宜蘭': {'宜蘭': ['冬山', '宜蘭']},

    '竹苗': {'新竹': ['新竹', '湖口', '竹東'],
           '苗栗': ['三義', '苗栗']},

    '花東': {'花蓮': ['花蓮'],
           '台東': ['臺東']},

    '北部離島': {'彭佳嶼': []},

    '西部離島': {'金門': ['金門'],
             '連江': ['馬祖'],
             '東吉嶼': [],
             '澎湖': ['馬公']},

    '雲嘉南': {'雲林': ['崙背', '斗六'],
            '台南': ['善化', '安南', '新營', '臺南'],
            '嘉義': ['嘉義', '新港', '朴子']},

    '高屏': {'高雄': ['仁武', '前金', '大寮', '小港', '左營', '林園', '楠梓', '美濃'],
           '屏東': ['屏東', '恆春', '潮州']}
}


local = '北部'
city = '台北'
site_list = pollution_site_map[local][city]  # ['中山', '古亭', '士林', '松山', '萬華']
target_site = '萬華'

training_year = ['2014', '2016']  # change format from   2014-2015   to   ['2014', '2015']
testing_year = ['2016', '2016']

training_duration = ['1/1', '10/31']
testing_duration = ['11/1', '12/31']
interval_hours = 24  # predict the label of average data of many hours later, default is 1
is_training = False
strategy = 'highest'  # highest or shift

# local = os.sys.argv[1]
# city = os.sys.argv[2]
# site_list = pollution_site_map[local][city]
# target_site = os.sys.argv[3]
#
# training_year = [os.sys.argv[4][:os.sys.argv[4].index('-')], os.sys.argv[4][os.sys.argv[4].index('-')+1:]]  # change format from   2014-2015   to   ['2014', '2015']
# testing_year = [os.sys.argv[5][:os.sys.argv[5].index('-')], os.sys.argv[5][os.sys.argv[5].index('-')+1:]]
#
# training_duration = [os.sys.argv[6][:os.sys.argv[6].index('-')], os.sys.argv[6][os.sys.argv[6].index('-')+1:]]
# testing_duration = [os.sys.argv[7][:os.sys.argv[7].index('-')], os.sys.argv[7][os.sys.argv[7].index('-')+1:]]
# interval_hours = int(os.sys.argv[8])  # predict the label of average data of many hours later, default is 1
# is_training = True if (os.sys.argv[9] == 'True' or os.sys.argv[9] == 'true') else False  # True False
# strategy = os.sys.argv[10]

# clear redundancy work
if training_year[0] == training_year[1]:
    training_year.pop(1)
if testing_year[0] == testing_year[1]:
    testing_year.pop(1)
else:
    input('The range of testing year should not more than one year or crossing the bound of years.')

# checking years
rangeofYear = int(training_year[-1])-int(training_year[0])
for i in range(rangeofYear):
    if not(str(i+int(training_year[0])) in training_year):
        training_year.insert(i, str(i+int(training_year[0])))

# Training Parameters
# WIND_DIREC is a specific feature, that need to be processed, and it can only be element of input vector now.
pollution_kind = ['PM2.5', 'O3', 'SO2', 'CO', 'NO2', 'WIND_SPEED', 'WIND_DIREC']  # 'SO2', 'CO', 'NO2' # 'AMB_TEMP', 'RH'
target_kind = 'PM2.5'
data_update = False
epochs = 50
batch_size = 64
seed = 0


# Network Parameters
input_size = (len(site_list)*len(pollution_kind)+len(site_list)) if 'WIND_DIREC' in pollution_kind else (len(site_list)*len(pollution_kind))
freq_input_size = len(site_list)*len(pollution_kind)
time_steps = 24 * 30
fourier_time_range = 24 * 15
fourier_time_shift = 24 * 5

# hidden_size = 20
num_classes = 1

testing_month = testing_duration[0][:testing_duration[0].index('/')]
training_begining = training_duration[0][:training_duration[0].index('/')]
training_deadline = training_duration[-1][:training_duration[-1].index('/')]
model_folder = root_path + "model/%s/%s/%sh/" % (local, city, interval_hours)
filename = ("CNN_%s_%s_training_%s_m%s_to_%s_m%s_interval_%s_%s"
            % (strategy, target_site, training_year[0], training_begining, training_year[-1], training_deadline,
               interval_hours, target_kind))
print(filename)
print('Training for %s/%s to %s/%s' % (training_year[0], training_duration[0], training_year[-1], training_duration[-1]))
print('Testing for %s/%s to %s/%s' % (testing_year[0], testing_duration[0], testing_year[-1], testing_duration[-1]))

if is_training:
    print('Training ..')
else:
    print('Testing ..')

# reading data
print('Reading data .. ')
start_time = time.time()
print('preparing training set ..')
X_train = read_data_sets(sites=site_list+[target_site], date_range=np.atleast_1d(training_year),
                         beginning=training_duration[0], finish=training_duration[-1],
                         feature_selection=pollution_kind, update=data_update)
X_train = missing_check(X_train)
Y_train = np.array(X_train)[:, -len(pollution_kind):]
Y_train = Y_train[:, pollution_kind.index(target_kind)]
X_train = np.array(X_train)[:, :-len(pollution_kind)]

print('preparing testing set ..')
X_test = read_data_sets(sites=site_list + [target_site], date_range=np.atleast_1d(testing_year),
                        beginning=testing_duration[0], finish=testing_duration[-1],
                        feature_selection=pollution_kind, update=data_update)
Y_test = np.array(X_test)[:, -len(pollution_kind):]
Y_test = Y_test[:, pollution_kind.index(target_kind)]
X_test = missing_check(np.array(X_test)[:, :-len(pollution_kind)])

final_time = time.time()
print('Reading data .. ok, ', end='')
time_spent_printer(start_time, final_time)

# print(len(X_train), 'train sequences')
# print(len(X_test), 'test sequences')

if (len(X_train) < time_steps) or (len(X_test) < time_steps):
    input('time_steps(%d) too long.' % time_steps)


# feature process
if 'WIND_DIREC' in pollution_kind:
    index_of_kind = pollution_kind.index('WIND_DIREC')
    length_of_kind_list = len(pollution_kind)
    len_of_sites_list = len(site_list)
    X_train = X_train.tolist()
    X_test = X_test.tolist()
    for i in range(len(X_train)):
        for j in range(len_of_sites_list):
            specific_index = index_of_kind + j * length_of_kind_list
            coordin = data_coordinate_angle(X_train[i].pop(specific_index+j))
            X_train[i].insert(specific_index + j, coordin[1])
            X_train[i].insert(specific_index + j, coordin[0])
            if i < len(X_test):
                coordin = data_coordinate_angle(X_test[i].pop(specific_index+j))
                X_test[i].insert(specific_index + j, coordin[1])
                X_test[i].insert(specific_index + j, coordin[0])
    X_train = np.array(X_train)
    X_test = np.array(X_test)
Y_test = np.array(Y_test, dtype=np.float)

# normalize
print('Normalize ..')

mean_y_train = np.mean(Y_train)
std_y_train = np.std(Y_train)
if not std_y_train:
    input("Denominator can't be 0.")
Y_train = [(y - mean_y_train) / std_y_train for y in Y_train]
print('mean_y_train: %f  std_y_train: %f' % (mean_y_train, std_y_train))

# fw = open(model_folder + "/%s_parameter.pickle" % target_site, 'wb')
# cPickle.dump(str(mean_y_train) + ',' +
#              str(std_y_train), fw)
# fw.close()

# --
print('Constructing time series data set ..', end='')
X_train = construct_time_steps(X_train[:-1], time_steps)
Y_train = Y_train[time_steps:]

X_test = construct_time_steps(X_test[:-1], time_steps)
Y_test = Y_test[time_steps:]
print('ok')


def ave(Y, interval_hours):
    reserve_hours = interval_hours - 1
    deadline = 0
    for i in range(len(Y)):
        # check the reserve data is enough or not
        if (len(Y) - i - 1) < reserve_hours:
            deadline = i
            break  # not enough
        for j in range(reserve_hours):
            Y[i] += Y[i + j + 1]
        Y[i] /= interval_hours
    if deadline:
        Y = Y[:deadline]
    return Y


def higher(Y, interval_hours):
    reserve_hours = 1  # choose the first n number of biggest
    if interval_hours > reserve_hours:
        deadline = 0
        for i in range(len(Y)):
            # check the reserve data is enough or not
            if (len(Y) - i) < interval_hours:
                deadline = i
                break  # not enough
            higher_list = []
            for j in range(interval_hours):
                if len(higher_list) < reserve_hours:
                    higher_list.append(Y[i + j])
                elif Y[i + j] > higher_list[0]:
                    higher_list[0] = Y[i + j]
                higher_list = sorted(higher_list)
            Y[i] = np.array(higher_list).sum()/reserve_hours
        if deadline:
            Y = Y[:deadline]
    return Y


Y_real = np.copy(Y_test)
if strategy == 'highest':
    Y_train = higher(Y_train, interval_hours)
    Y_test = higher(Y_test, interval_hours)
elif strategy == 'shift':
    Y_train = Y_train[interval_hours-1:]
    Y_test = Y_test[interval_hours-1:]
Y_real = Y_real[interval_hours - 1:]

train_seq_len = np.min([len(Y_train), len(X_train)])
test_seq_len = np.min([len(Y_test), len(X_test)])

X_train = X_train[:train_seq_len]
X_test = X_test[:test_seq_len]

Y_train = Y_train[:train_seq_len]
Y_test = Y_test[:test_seq_len]
Y_real = Y_real[:test_seq_len]

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')


# -- fourier transfer --
def time_domain_to_frequency_domain(time_tensor):
    freq_tensor = []
    for f_i in range(len(time_tensor)):
        freq_feature_matrix = []
        for f_j in range(len(pollution_kind)*len(site_list)):
            freq_feature_vector = np.array([])
            for f_k in range(((time_steps-fourier_time_range)/fourier_time_shift)+1):
                freq_feature_vector = np.concatenate((
                    freq_feature_vector,
                    np.real(
                        fft(time_tensor[f_i, f_k*fourier_time_shift:f_k*fourier_time_shift+fourier_time_range, f_j])
                    )
                ))
            freq_feature_matrix.append(freq_feature_vector)
        freq_tensor.append(np.array(freq_feature_matrix).T)
    return freq_tensor


start_time = time.time()
print('fourier transfer .. ')
print('for training data ..')
X_train = np.array(X_train)
X_train_freq = time_domain_to_frequency_domain(X_train)
print('for testing data ..')
X_test = np.array(X_test)
X_test_freq = time_domain_to_frequency_domain(X_test)
final_time = time.time()
print('fourier transfer .. ok, ', end='')
time_spent_printer(start_time, final_time)


# delete data which have missing values
i = 0
while i < len(Y_test):
    if not(Y_test[i] > -10000):  # check missing or not, if Y_test[i] is missing, then this command will return True
        Y_test = np.delete(Y_test, i, 0)
        Y_real = np.delete(Y_real, i, 0)
        X_test = np.delete(X_test, i, 0)
        X_test_freq = np.delete(X_test_freq, i, 0)
        i = -1
    i += 1
Y_test = np.array(Y_test, dtype=np.float)
Y_real = np.array(Y_real, dtype=np.float)

print('delete invalid testing data, remain ', len(X_test), 'test sequences')

X_train = np.array(X_train)
X_train_freq = np.array(X_train_freq)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
X_test_freq = np.array(X_test_freq)

np.random.seed(seed)
np.random.shuffle(X_train)
np.random.seed(seed)
np.random.shuffle(Y_train)

# --
freq_time_step = (((time_steps-fourier_time_range)/fourier_time_shift)+1)*fourier_time_range
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, time_steps, input_size)
    X_test = X_test.reshape(X_test.shape[0], 1, time_steps, input_size)
    input_shape = (1, time_steps, input_size)

    X_train_freq = X_train_freq.reshape(X_train_freq.shape[0], 1, freq_time_step, freq_input_size)
    X_test_freq = X_test_freq.reshape(X_test_freq.shape[0], 1, freq_time_step, freq_input_size)
    freq_input_shape = (1, freq_time_step, freq_input_size)
else:
    X_train = X_train.reshape(X_train.shape[0], time_steps, input_size, 1)
    X_test = X_test.reshape(X_test.shape[0], time_steps, input_size, 1)
    input_shape = (time_steps, input_size, 1)

    X_train_freq = X_train_freq.reshape(X_train_freq.shape[0], freq_time_step, freq_input_size, 1)
    X_test_freq = X_test_freq.reshape(X_test_freq.shape[0], freq_time_step, freq_input_size, 1)
    freq_input_shape = (freq_time_step, freq_input_size, 1)

print('input_shape: ', input_shape)
print('freq_input_shape: ', freq_input_shape)

# -- model --
print('Building model ..')

if is_training:
    # model = Sequential()
    # model.add(Conv2D(128, kernel_size=(3, input_size),
    #                  activation='relu',
    #                  input_shape=input_shape))
    # model.add(Conv2D(64, (3, 1), activation='relu'))
    # # model.add(MaxPooling2D(pool_size=(2, 1)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
    #                              gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes))

    # time domain
    model_input_time = Input(shape=input_shape, dtype='float32')
    cnn_layer_time = []
    for i in [6, 12, 18, 24]:
        first_layer_time = Conv2D(8, kernel_size=(i, input_size), activation='relu')(model_input_time)
        second_layer_time = Conv2D(4, kernel_size=(i, 1), activation='relu')(first_layer_time)
        cnn_layer_time.append(Flatten()(second_layer_time))
    cnn_model_time = concatenate(cnn_layer_time)
    cnn_model_time = Dropout(0.25)(cnn_model_time)

    cnn_model_time = BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
                                        gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(cnn_model_time)
    cnn_model_time = Dense(4, activation='relu')(cnn_model_time)
    cnn_model_time = Dropout(0.5)(cnn_model_time)

    highway_time = Flatten()(model_input_time)
    highway_time = BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
                                      gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(highway_time)
    cnn_model_time = concatenate([highway_time, cnn_model_time])
    # cnn_model_time = Dense(num_classes)(cnn_model_time)
    cnn_model_time = Dense(2, activation='relu')(cnn_model_time)

    # frequency domain
    model_input_freq = Input(shape=freq_input_shape, dtype='float32')
    first_layer_freq = Conv2D(8, kernel_size=(i, freq_input_size), activation='relu')(model_input_freq)
    cnn_model_freq = Flatten()(first_layer_freq)
    cnn_model_freq = Dropout(0.25)(cnn_model_freq)

    cnn_model_freq = BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
                                        gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(cnn_model_freq)
    cnn_model_freq = Dense(4, activation='relu')(cnn_model_freq)
    cnn_model_freq = Dropout(0.5)(cnn_model_freq)

    cnn_model_freq = Dense(2, activation='relu')(cnn_model_freq)
    cnn_model_freq = Dropout(0.5)(cnn_model_freq)

    # concatenation
    cnn_model = concatenate([cnn_model_time, cnn_model_freq])

    cnn_model = BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
                                   gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(cnn_model)
    cnn_model = Dense(num_classes)(cnn_model)

    # --

    model = Model(inputs=[model_input_time, model_input_freq], outputs=cnn_model)
    model.compile(loss=keras.losses.mean_squared_error,
                  # optimizer=keras.optimizers.Adadelta(),
                  optimizer='nadam',
                  metrics=['accuracy'])

    print('Start training ..')

    # --
    start_time = time.time()

    model.fit([X_train, X_train_freq], Y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=([X_test, X_test_freq], ((Y_test-mean_y_train)/std_y_train)))

    final_time = time.time()
    print('Training .. ok, ', end='')
    time_spent_printer(start_time, final_time)
    # --

    score = model.evaluate([X_test, X_test_freq], ((Y_test-mean_y_train)/std_y_train), verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save(model_folder + filename, overwrite=True)
    model.save(model_folder + 'backup/' + filename, overwrite=True)
    print('model saved: ', filename)
else:
    model = keras.models.load_model(model_folder + filename)

    # fr = open(model_folder + "%s_parameter.pickle" % target_site, 'rb')
    # [mean_y_train, std_y_train] = (cPickle.load(fr)).split(',')
    # mean_y_train = float(mean_y_train)
    # std_y_train = float(std_y_train)
    # fr.close()

print('prediction ..')

pred = model.predict([X_test, X_test_freq])
pred = mean_y_train + std_y_train * pred

print('rmse: %.5f' % (np.mean((Y_test - pred.reshape([len(Y_test)]))**2, 0)**0.5))

# with open(root_path + 'result/' + filename + '.ods', 'wt') as f:
#     f.write('rmse: %f' % (np.sqrt(np.mean((Y_test - pred) ** 2))))

plt.plot(np.arange(len(pred)), Y_real[:len(pred)], c='lightblue')
plt.plot(np.arange(len(pred)), Y_test[:len(pred)], c='gray')
plt.plot(np.arange(len(pred)), pred, color='pink')
plt.xticks(np.arange(0, len(pred), 24))
plt.yticks(np.arange(0, max(Y_test), 10))
plt.grid(True)
plt.rc('axes', labelsize=4)
plt.savefig(root_path + 'result/' + filename + '.png')
# plt.show()

# -- check overfitting --

# train_pred = model.predict([X_train[-800:], X_train_freq[-800:]])
# train_pred = mean_y_train + std_y_train * train_pred
# train_pred_target = mean_y_train + std_y_train * Y_train[-800:]
# plt.plot(np.arange(len(train_pred)), train_pred, c='gray')
# plt.plot(np.arange(len(train_pred)), train_pred_target, color='pink')
# plt.xticks(np.arange(0, len(train_pred), 24))
# plt.yticks(np.arange(0, max(train_pred), 10))
# plt.grid(True)
# plt.rc('axes', labelsize=4)
# plt.show()
