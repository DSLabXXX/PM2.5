# _*_ coding: utf-8 _*_

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import time
import cPickle
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import h5py

import keras
from keras.layers.core import Dense, Dropout, Reshape, Flatten
from keras.layers import Input, Conv2D, concatenate
from keras.layers.recurrent import LSTM
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K


from reader import read_data_sets, construct_time_steps, construct_second_time_steps
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
    '中部': {'台中': ['大里', '忠明', '沙鹿', '西屯', '豐原'],  # 5
           '南投': ['南投', '竹山'],  # 2
           '彰化': ['二林', '彰化']},  # 2

    '北部': {'台北': ['中山', '古亭', '士林', '松山', '萬華'],  # 5
           '新北': ['土城', '新店', '新莊', '板橋', '林口', '汐止', '菜寮', '萬里'],  # 8
           '基隆': ['基隆'],  # 1
           '桃園': ['大園', '平鎮', '桃園', '龍潭']}, # 4

    '宜蘭': {'宜蘭': ['冬山', '宜蘭']},  # 2

    '竹苗': {'新竹': ['新竹', '湖口', '竹東'],  # 3
           '苗栗': ['三義', '苗栗']},  # 2

    '花東': {'花蓮': ['花蓮'],  # 1
           '台東': ['臺東']},  # 1

    '北部離島': {'彭佳嶼': []},

    '西部離島': {'金門': ['金門'], # 1
             '連江': ['馬祖'],  # 1
             '東吉嶼': [],
             '澎湖': ['馬公']},  # 1

    '雲嘉南': {'雲林': ['崙背', '斗六', '竹山'],  # 3
            '台南': ['善化', '安南', '新營', '臺南'],  # 4
            '嘉義': ['嘉義', '新港', '朴子']},  # 3

    '高屏': {'高雄': ['仁武', '前金', '大寮', '小港', '左營', '林園', '楠梓', '美濃'],  # 8
           '屏東': ['屏東', '恆春', '潮州']}  # 3
}


def h5file_check(model_file):
    with h5py.File(model_file, 'a') as f:
        if 'optimizer_weights' in f.keys():
            del f['optimizer_weights']


# high_alert = 53.5
# low_alert = 35.5

local = '北部'
city = '台北'
site_list = pollution_site_map[local][city]  # ['中山', '古亭', '士林', '松山', '萬華']
target_site = '萬華'

training_year = ['2014', '2016']  # change format from   2014-2015   to   ['2014', '2015']
testing_year = ['2016', '2016']

training_duration = ['1/1', '10/31']
testing_duration = ['11/1', '12/31']
interval_hours = 24  # predict the label of average data of many hours later, default is 1
is_training = True

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

target_kind = 'PM2.5'

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
pollution_kind = ['PM2.5', 'O3', 'SO2', 'CO', 'NO2', 'WIND_SPEED', 'WIND_DIREC']  # , 'AMB_TEMP', 'RH'
data_update = False
epochs = 50
batch_size = 512
p_dense = 0.5
regularizer = float('1e-6')
seed = 0

# Network Parameters
input_size = (len(site_list)*len(pollution_kind)+len(site_list)) if 'WIND_DIREC' in pollution_kind else (len(site_list)*len(pollution_kind))

layer1_time_steps = 24  # 24 hours a day for cnn
layer2_time_steps = 28  # 7 days for rnn

cnn_hidden1_size = 32
cnn_hidden2_size = 16
# cnn_hidden3_size = 4

rnn_hidden1_size = 16
rnn_hidden2_size = 8
# rnn_hidden3_size = 8

output_layer1_size = 4
output_layer2_size = 1

fourier_time_range = 24 * 14
fourier_time_shift = 24 * 7
freq_input_size = len(site_list)*len(pollution_kind)
total_time_steps = layer1_time_steps*layer2_time_steps

if total_time_steps < fourier_time_range:
    print('time steps error(1)')
    exit()
if ((total_time_steps-fourier_time_range) % fourier_time_shift) != 0:
    print('time steps error(2)')
    exit()

# --

testing_month = testing_duration[0][:testing_duration[0].index('/')]
folder = root_path+"model/%s/%s/%sh/" % (local, city, interval_hours)
training_begining = training_duration[0][:training_duration[0].index('/')]
training_deadline = training_duration[-1][:training_duration[-1].index('/')]
print('site: %s' % target_site)
print('Training for %s/%s to %s/%s' % (training_year[0], training_duration[0], training_year[-1], training_duration[-1]))
print('Testing for %s/%s to %s/%s' % (testing_year[0], testing_duration[0], testing_year[-1], testing_duration[-1]))
print('Target: %s' % target_kind)


# for interval
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


# for interval
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
            Y[i] = np.array(higher_list).sum() / reserve_hours
        if deadline:
            Y = Y[:deadline]
    return Y


# if True:  # is_training:
# reading data
print('Reading data .. ')
start_time = time.time()
initial_time = time.time()
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

# if (len(X_train) < time_steps) or (len(X_test) < time_steps):
#     input('time_steps(%d) too long.' % time_steps)

# normalize
print('Normalize ..')
mean_X_train = np.mean(X_train, axis=0)
std_X_train = np.std(X_train, axis=0)
if 0 in std_X_train:
    input("Denominator can't be 0.")
X_train = np.array([(x_train-mean_X_train)/std_X_train for x_train in X_train])
X_test = np.array([(x_test-mean_X_train)/std_X_train for x_test in X_test])

mean_y_train = np.mean(Y_train)
std_y_train = np.std(Y_train)
if not std_y_train:
    input("Denominator can't be 0.")
Y_train = [(y - mean_y_train) / std_y_train for y in Y_train]
print('mean_y_train: %f  std_y_train: %f' % (mean_y_train, std_y_train))

fw = open(folder + "%s_parameter_%s.pickle" % (target_site, target_kind), 'wb')
cPickle.dump(str(mean_X_train) + ',' +
             str(std_X_train) + ',' +
             str(mean_y_train) + ',' +
             str(std_y_train), fw)
fw.close()

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
            coordin = data_coordinate_angle((X_train[i].pop(specific_index+j))*std_X_train[specific_index]+mean_X_train[specific_index])
            X_train[i].insert(specific_index + j, coordin[1])
            X_train[i].insert(specific_index + j, coordin[0])
            if i < len(X_test):
                coordin = data_coordinate_angle((X_test[i].pop(specific_index+j))*std_X_train[specific_index]+mean_X_train[specific_index])
                X_test[i].insert(specific_index + j, coordin[1])
                X_test[i].insert(specific_index + j, coordin[0])
    X_train = np.array(X_train)
    X_test = np.array(X_test)
Y_test = np.array(Y_test, dtype=np.float)

# --

print('Constructing time series data set ..')


def time_domain_to_frequency_domain(time_tensor):
    freq_tensor = []
    for f_i in range(len(time_tensor)):
        freq_feature_matrix = []
        for f_j in range(len(pollution_kind)*len(site_list)):
            freq_feature_vector = np.array([])
            for f_k in range(((total_time_steps-fourier_time_range)/fourier_time_shift)+1):
                freq_feature_vector = np.concatenate((
                    freq_feature_vector,
                    np.real(
                        fft(time_tensor[f_i, f_k*fourier_time_shift:f_k*fourier_time_shift+fourier_time_range, f_j])
                    )
                ))
            freq_feature_matrix.append(freq_feature_vector)
        freq_tensor.append(np.array(freq_feature_matrix).T)
    return freq_tensor

# for frequence
X_train_freq = construct_time_steps(X_train[:-1], layer1_time_steps*layer2_time_steps)
X_test_freq = construct_time_steps(X_test[:-1], layer1_time_steps*layer2_time_steps)


# for layer 1
X_train = construct_time_steps(X_train[:-1], layer1_time_steps)
X_test = construct_time_steps(X_test[:-1], layer1_time_steps)

Y_train = Y_train[layer1_time_steps:]
Y_test = Y_test[layer1_time_steps:]


# for layer 2
X_train = construct_second_time_steps(X_train, layer1_time_steps, layer2_time_steps)
X_test = construct_second_time_steps(X_test, layer1_time_steps, layer2_time_steps)

Y_train = Y_train[layer1_time_steps*(layer2_time_steps-1):]
Y_test = Y_test[layer1_time_steps*(layer2_time_steps-1):]

# --

Y_real = np.copy(Y_test)

# Y_train = higher(Y_train, interval_hours)
# Y_test = higher(Y_test, interval_hours)
Y_train = Y_train[interval_hours-1:]
Y_test = Y_test[interval_hours-1:]
Y_real = Y_real[interval_hours-1:]

train_seq_len = np.min([len(Y_train), len(X_train), len(X_train_freq)])
test_seq_len = np.min([len(Y_test), len(X_test), len(X_test_freq)])

# print(len(X_train), 'train sequences')
# print(len(X_test), 'test sequences')

X_train = X_train[:train_seq_len]
X_test = X_test[:test_seq_len]
X_train_freq = X_train_freq[:train_seq_len]
X_test_freq = X_test_freq[:test_seq_len]

Y_train = Y_train[:train_seq_len]
Y_test = Y_test[:test_seq_len]
Y_real = Y_real[:test_seq_len]


# ----------------------------------------- fourier transfer -----------------------------------------

start_time = time.time()
print('fourier transfer .. ')
print('for training data ..')
X_train_freq = np.array(X_train_freq)
X_train_freq = time_domain_to_frequency_domain(X_train_freq)
print('for testing data ..')
X_test_freq = np.array(X_test_freq)
X_test_freq = time_domain_to_frequency_domain(X_test_freq)
final_time = time.time()
print('fourier transfer .. ok, ', end='')
time_spent_printer(start_time, final_time)

# ----------------------------------------------------------------------------------------------------


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

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

# --

X_train = np.array(X_train)
X_test = np.array(X_test)
X_train_freq = np.array(X_train_freq)
X_test_freq = np.array(X_test_freq)

Y_train = np.array(Y_train)

# np.random.seed(seed)
# np.random.shuffle(Y_train)
#
# np.random.seed(seed)
# np.random.shuffle(X_train)

# else:  # is_training = false
#     # mean and std
#     fr = open(folder + "%s_parameter_%s.pickle" % (target_site, target_kind), 'rb')
#     [mean_X_train, std_X_train, mean_y_train, std_y_train] = (cPickle.load(fr)).split(',')
#     mean_X_train = mean_X_train.replace('[', '').replace(']', '').replace('\n', '').split(' ')
#     while '' in mean_X_train:
#         mean_X_train.pop(mean_X_train.index(''))
#     mean_X_train = np.array(mean_X_train, dtype=np.float)
#     std_X_train = std_X_train.replace('[', '').replace(']', '').replace('\n', '').split(' ')
#     while '' in std_X_train:
#         std_X_train.pop(std_X_train.index(''))
#     std_X_train = np.array(std_X_train, dtype=np.float)
#     mean_y_train = float(mean_y_train)
#     std_y_train = float(std_y_train)
#     fr.close()
#
#     # reading data
#     print('preparing testing set ..')
#     X_test = read_data_sets(sites=site_list + [target_site], date_range=np.atleast_1d(testing_year),
#                             beginning=testing_duration[0], finish=testing_duration[-1],
#                             feature_selection=pollution_kind, update=data_update)
#     Y_test = np.array(X_test)[:, -len(pollution_kind):]
#     Y_test = Y_test[:, pollution_kind.index(target_kind)]
#     X_test = missing_check(np.array(X_test)[:, :-len(pollution_kind)])
#
#     # normalize
#     print('Normalize ..')
#     if 0 in std_X_train:
#         input("Denominator can't be 0.")
#     X_test = np.array([(x_test-mean_X_train)/std_X_train for x_test in X_test])
#
#     # feature process
#     if 'WIND_DIREC' in pollution_kind:
#         index_of_kind = pollution_kind.index('WIND_DIREC')
#         length_of_kind_list = len(pollution_kind)
#         len_of_sites_list = len(site_list)
#         X_test = X_test.tolist()
#         for i in range(len(X_test)):
#             for j in range(len_of_sites_list):
#                 specific_index = index_of_kind + j * length_of_kind_list
#                 coordin = data_coordinate_angle(
#                     (X_test[i].pop(specific_index + j)) * std_X_train[specific_index] + mean_X_train[
#                         specific_index])
#                 X_test[i].insert(specific_index + j, coordin[1])
#                 X_test[i].insert(specific_index + j, coordin[0])
#         X_test = np.array(X_test)
#     Y_test = np.array(Y_test, dtype=np.float)
#
#     # --
#
#     print('Constructing time series data set ..')
#     # for layer 1 -----------------------------------------------------------------------------------------------------------------
#     X_test = construct_time_steps(X_test[:-1], layer1_time_steps)
#     Y_test = Y_test[layer1_time_steps:]
#
#     # for layer 2
#     X_test = construct_second_time_steps(X_test, layer1_time_steps, layer2_time_steps)
#     Y_test = Y_test[layer1_time_steps*(layer2_time_steps-1):]
#
#     # --
#     Y_real = np.copy(Y_test)
#
#     # Y_test = higher(Y_test, interval_hours)
#     Y_test = Y_test[interval_hours - 1:]
#     Y_real = Y_real[interval_hours - 1:]
#
#     test_seq_len = np.min([len(Y_test), len(X_test)])
#
#     print(len(X_test), 'test sequences')
#
#     X_test = X_test[:test_seq_len]
#
#     Y_test = Y_test[:test_seq_len]
#     Y_real = Y_real[:test_seq_len]
#
#     # delete data which have missing values
#     i = 0
#     while i < len(Y_test):
#         if not (Y_test[i] > -10000):  # check missing or not, if Y_test[i] is missing, then this command will return True
#             Y_test = np.delete(Y_test, i, 0)
#             Y_real = np.delete(Y_real, i, 0)
#             X_test = np.delete(X_test, i, 0)
#             i = -1
#         i += 1
#     Y_test = np.array(Y_test, dtype=np.float)
#
#     print('delete invalid testing data, remain ', len(X_test), 'test sequences')
#
#     # --
#
#     X_test = np.array(X_test)


# -- rnn --
print('- rnn -')

filename = ("RCNN_%s_training_%s_m%s_to_%s_m%s_interval_%s_%s"
            % (target_site, training_year[0], training_begining, training_year[-1], training_deadline, interval_hours, target_kind))
print(filename)


# --

print('Build rnn model...')
start_time = time.time()

# input layer
# input_shape = (layer1_time_steps, input_size)
freq_time_steps = (((total_time_steps-fourier_time_range)/fourier_time_shift)+1)*fourier_time_range
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, layer2_time_steps, layer1_time_steps, input_size)
    X_test = X_test.reshape(X_test.shape[0], 1, layer2_time_steps, layer1_time_steps, input_size)
    input_shape = (1, layer1_time_steps, input_size)

    X_train_freq = X_train_freq.reshape(X_train_freq.shape[0], 1, freq_time_steps, freq_input_size)
    X_test_freq = X_test_freq.reshape(X_test_freq.shape[0], 1, freq_time_steps, freq_input_size)
    freq_input_shape = (1, freq_time_steps, freq_input_size)
else:
    X_train = X_train.reshape(X_train.shape[0], layer2_time_steps, layer1_time_steps, input_size, 1)
    X_test = X_test.reshape(X_test.shape[0], layer2_time_steps, layer1_time_steps, input_size, 1)
    input_shape = (layer1_time_steps, input_size, 1)

    X_train_freq = X_train_freq.reshape(X_train_freq.shape[0], freq_time_steps, freq_input_size, 1)
    X_test_freq = X_test_freq.reshape(X_test_freq.shape[0], freq_time_steps, freq_input_size, 1)
    freq_input_shape = (freq_time_steps, freq_input_size, 1)

rnn_model_input = list()
for i in range(layer2_time_steps):
    rnn_model_input.append(Input(shape=input_shape, dtype='float32'))

# -- cnn layer --
model_layer1 = list()

for i in range(layer2_time_steps):
    model_layer1.append(BatchNormalization(beta_regularizer=None, epsilon=0.001, beta_initializer="zero",
                                           gamma_initializer="one", weights=None, gamma_regularizer=None,
                                           momentum=0.99, axis=-1)(rnn_model_input[i]))

    flatten_cnn_layer = []
    for j in [3, 6, 9, 12]:
        first_cnn_layer = Conv2D(cnn_hidden1_size, kernel_size=(j, input_size), activation='relu')(model_layer1[i])
        second_cnn_layer = Conv2D(cnn_hidden2_size, kernel_size=(j, 1), activation='relu')(first_cnn_layer)
        flatten_cnn_layer.append(Flatten()(second_cnn_layer))

    output_cnn_layer = concatenate(flatten_cnn_layer)
    model_layer1[i] = Dropout(p_dense)(output_cnn_layer)

# -- rnn layer --
# layer 1
model_layer2 = concatenate(model_layer1)
model_layer2 = Reshape((layer2_time_steps, -1))(model_layer2)

model_layer2 = BatchNormalization(beta_regularizer=None, epsilon=0.001, beta_initializer="zero", gamma_initializer="one",
                                  weights=None, gamma_regularizer=None, momentum=0.99, axis=-1)(model_layer2)
model_layer2 = LSTM(rnn_hidden1_size, kernel_regularizer=l2(regularizer), recurrent_regularizer=l2(regularizer),
                    bias_regularizer=l2(regularizer), recurrent_dropout=0.5, return_sequences=True)(
    model_layer2)

model_layer2 = Dropout(p_dense)(model_layer2)

# layer 2
model_layer2 = model_layer2
model_layer2 = BatchNormalization(beta_regularizer=None, epsilon=0.001, beta_initializer="zero", gamma_initializer="one",
                                  weights=None, gamma_regularizer=None, momentum=0.99, axis=-1)(model_layer2)
model_layer2 = LSTM(rnn_hidden2_size, kernel_regularizer=l2(regularizer), recurrent_regularizer=l2(regularizer),
                    bias_regularizer=l2(regularizer), recurrent_dropout=0.5)(
    model_layer2)

model_layer2 = Dropout(p_dense)(model_layer2)
# # layer 3
# model_layer2 = model_layer2
# model_layer2 = BatchNormalization(beta_regularizer=None, epsilon=0.001, beta_initializer="zero", gamma_initializer="one",
#                                   weights=None, gamma_regularizer=None, momentum=0.99, axis=-1)(model_layer2)
# model_layer2 = LSTM(rnn_hidden2_size, kernel_regularizer=l2(regularizer), recurrent_regularizer=l2(regularizer),
#                     bias_regularizer=l2(regularizer), recurrent_dropout=0.5)(
#     model_layer2)
#
# model_layer2 = Dropout(p_dense)(model_layer2)

# -- frequence layer --
model_input_freq = Input(shape=freq_input_shape, dtype='float32')
first_layer_freq = Conv2D(8, kernel_size=(i, freq_input_size), activation='relu')(model_input_freq)
model_freq = Flatten()(first_layer_freq)
model_freq = Dropout(0.25)(model_freq)

model_freq = BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
                                    gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(model_freq)
model_freq = Dense(4, activation='relu')(model_freq)
model_freq = Dropout(0.5)(model_freq)

model_freq = Dense(2, activation='relu')(model_freq)
model_freq = Dropout(0.5)(model_freq)


# -- output layer --
output_layer = concatenate([model_layer2, model_freq])  # concatenation
# output_layer = model_layer2

output_layer = BatchNormalization(beta_regularizer=None, epsilon=0.001, beta_initializer="zero", gamma_initializer="one",
                                  weights=None, gamma_regularizer=None, momentum=0.99, axis=-1)(output_layer)
output_layer = Dense(output_layer1_size, kernel_regularizer=l2(regularizer), bias_regularizer=l2(regularizer))(output_layer)
output_layer = Dropout(p_dense)(output_layer)

output_layer = BatchNormalization(beta_regularizer=None, epsilon=0.001, beta_initializer="zero", gamma_initializer="one",
                                  weights=None, gamma_regularizer=None, momentum=0.99, axis=-1)(output_layer)
output_layer = Dense(output_layer2_size, kernel_regularizer=l2(regularizer), bias_regularizer=l2(regularizer))(output_layer)

# --
model_input = rnn_model_input
model_input.append(model_input_freq)
rnn_model = Model(inputs=rnn_model_input, outputs=output_layer)
rnn_model.compile(loss=keras.losses.mean_squared_error,
                  optimizer='adam',
                  metrics=['accuracy'])

# optimiser = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=False)
# optimiser = 'adam'
# rnn_model.compile(loss='mean_squared_error', optimizer=optimiser)

final_time = time.time()
time_spent_printer(start_time, final_time)


if is_training:
    print("Train...")
    start_time = time.time()

    train_input_list = [X_train[:, i, :, :] for i in range(layer2_time_steps)]
    train_input_list.append(X_train_freq)

    test_input_list = [X_test[:, i, :, :] for i in range(layer2_time_steps)]
    test_input_list.append(X_test_freq)

    rnn_model.fit(train_input_list, Y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(test_input_list,
                                   ((Y_test - mean_y_train) / std_y_train)))

    # Potentially save weights
    # rnn_model.save_weights(folder + filename, overwrite=True)
    rnn_model.save(folder + filename, overwrite=True)

    final_time = time.time()
    time_spent_printer(start_time, final_time)
    print('model saved: ', filename)

else:
    print('loading model ..')
    # print('loading model from %s' % (folder + filename + ".hdf5"))
    # rnn_model.load_weights(folder + filename)
    try:
        rnn_model = keras.models.load_model(folder + filename)
    except:
        h5file_check(folder + filename)
        rnn_model = keras.models.load_model(folder + filename)


test_input_list = [X_test[:, i, :, :] for i in range(layer2_time_steps)]
test_input_list.append(X_test_freq)


rnn_pred = rnn_model.predict(test_input_list)
final_time = time.time()
time_spent_printer(start_time, final_time)
print('rmse(rnn): %.5f' % (np.mean((np.atleast_2d(Y_test).T - (mean_y_train + std_y_train * rnn_pred))**2, 0)**0.5))

pred = mean_y_train + std_y_train * rnn_pred

plt.plot(np.arange(len(pred)), Y_real[:len(pred)], c='gray')
plt.plot(np.arange(len(pred)), Y_test[:len(pred)], c='mediumaquamarine')
plt.plot(np.arange(len(pred)), pred, color='pink')
plt.xticks(np.arange(0, len(pred), 24))
plt.yticks(np.arange(0, max(Y_test), 10))
plt.grid(True)
plt.rc('axes', labelsize=4)
plt.savefig(root_path + 'result/' + filename + '.png')
plt.show()

# -- check overfitting --

train_pred = rnn_model.predict([X_train[:, i, :, :][-600:] for i in range(layer2_time_steps)])
train_pred = mean_y_train + std_y_train * train_pred
train_pred_target = mean_y_train + std_y_train * Y_train[-600:]
plt.plot(np.arange(len(train_pred)), train_pred_target, color='mediumaquamarine')
plt.plot(np.arange(len(train_pred)), train_pred, c='pink')
plt.xticks(np.arange(0, len(train_pred), 24))
plt.yticks(np.arange(0, max(train_pred), 10))
plt.grid(True)
plt.rc('axes', labelsize=4)
plt.savefig(root_path + 'result/training_processing.png')
plt.show()
