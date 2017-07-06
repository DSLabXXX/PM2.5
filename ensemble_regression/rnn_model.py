# _*_ coding: utf-8 _*_

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import time
# import cPickle
# import os
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# import h5py
import keras
# from keras.optimizers import SGD, RMSprop, Adagrad
# from keras.models import Sequential
from keras import backend as K
from keras.layers.core import Dense, Dropout, Flatten, Reshape
# from keras.layers.embeddings import Embedding
from keras.layers import Input, Conv2D, concatenate
from keras.layers.recurrent import LSTM  # , GRU, SimpleRNN
from keras.regularizers import l2
# from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping

from reader import read_data_sets, concatenate_time_steps, construct_time_steps, construct_second_time_steps
from missing_value_processer import missing_check
from feature_processor import data_coordinate_angle
from config import root, site_map

root_path = root()


def time_spent_printer(start_time, final_time):
    spent_time = final_time - start_time
    print('totally spent ', end='')
    print(int(spent_time / 3600), 'hours ', end='')
    print(int((int(spent_time) % 3600) / 60), 'minutes ', end='')
    print((int(spent_time) % 3600) % 60, 'seconds')


# def h5file_check(model_file):
#     with h5py.File(model_file, 'a') as f:
#         if 'optimizer_weights' in f.keys():
#             del f['optimizer_weights']


pollution_site_map = site_map()


# high_alert = 53.5
# low_alert = 35.5


# local = os.sys.argv[1]
# city = os.sys.argv[2]
# target_site = os.sys.argv[3]
#
# training_year = [os.sys.argv[4][:os.sys.argv[4].index('-')], os.sys.argv[4][os.sys.argv[4].index('-')+1:]]  # change format from   2014-2015   to   ['2014', '2015']
# testing_year = [os.sys.argv[5][:os.sys.argv[5].index('-')], os.sys.argv[5][os.sys.argv[5].index('-')+1:]]
#
# training_duration = [os.sys.argv[6][:os.sys.argv[6].index('-')], os.sys.argv[6][os.sys.argv[6].index('-')+1:]]
# testing_duration = [os.sys.argv[7][:os.sys.argv[7].index('-')], os.sys.argv[7][os.sys.argv[7].index('-')+1:]]
# interval_hours = int(os.sys.argv[8])  # predict the label of average data of many hours later, default is 1
# is_training = True if (os.sys.argv[9] == 'True' or os.sys.argv[9] == 'true') else False  # True False


local = '北部'
city = '台北'
target_site = '萬華'

training_year = ['2014', '2016']  # change format from   2014-2015   to   ['2014', '2015']
testing_year = ['2016', '2016']

training_duration = ['1/1', '10/31']
testing_duration = ['11/15', '12/31']
interval_hours = 24  # predict the label of average data of many hours later, default is 1
is_training = True


# site_list = pollution_site_map[local][city]  # ['中山', '古亭', '士林', '松山', '萬華']
# --
site_list = list()
for i in pollution_site_map:
    for j in pollution_site_map[i]:
        try:
            site_list += [pollution_site_map[i][j][0]]
        except:
            site_list += []

if target_site not in site_list:
    site_list.append(target_site)
# --

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
pollution_kind = ['PM2.5', 'O3', 'SO2', 'CO', 'NO2', 'AMB_TEMP', 'RH', 'WIND_SPEED', 'WIND_DIREC']  # , 'AMB_TEMP', 'RH'
# pollution_group = list()
# pollution_group.append(['O3', 'SO2', 'WIND_SPEED', 'WIND_DIREC'])
# pollution_group.append(['O3', 'CO', 'WIND_SPEED', 'WIND_DIREC'])
# pollution_group.append(['O3', 'NO2', 'WIND_SPEED', 'WIND_DIREC'])
# pollution_group.append(['PM2.5', 'WIND_SPEED', 'WIND_DIREC'])
# pollution_group.append(pollution_kind)

feature_kind_shift = 6  # 'day of year', 'day of week' and 'time of day' respectively use two dimension


data_update = False
dropout = 0.5
regularizer = float('1e-6')
batch_size = 256
epoch = 50
seed = 0
recurrent_dropout = 0.5

# Network Parameters
layer1_time_steps = 24  # 24 hours a day
layer2_time_steps = 14  # 7 days

# rnn
hidden_size1 = 8
hidden_size2 = 16
# hidden_size3 = 4

# cnn
kernel_num_1 = 8
kernel_num_2 = 4
cnn_hidden_size1 = 8

output_size = 1

testing_month = testing_duration[0][:testing_duration[0].index('/')]
folder = root_path+"model/%s/%s/%sh/" % (local, city, interval_hours)
training_begining = training_duration[0][:training_duration[0].index('/')]
training_deadline = training_duration[-1][:training_duration[-1].index('/')]
print('site: %s' % target_site)
# print('site list: ', site_list)
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

raw_data_train = read_data_sets(sites=site_list+[target_site], date_range=np.atleast_1d(training_year),
                                beginning=training_duration[0], finish=training_duration[-1],
                                feature_selection=pollution_kind, update=data_update)
raw_data_train = missing_check(raw_data_train)
Y_train = np.array(raw_data_train)[:, -len(pollution_kind):]
Y_train = Y_train[:, pollution_kind.index(target_kind)]
raw_data_train = np.array(raw_data_train)[:, :-len(pollution_kind)]

print('preparing testing set ..')

raw_data_test = read_data_sets(sites=site_list + [target_site], date_range=np.atleast_1d(testing_year),
                               beginning=testing_duration[0], finish=testing_duration[-1],
                               feature_selection=pollution_kind, update=data_update)
Y_test = np.array(raw_data_test)[:, -len(pollution_kind):]
Y_test = Y_test[:, pollution_kind.index(target_kind)]
raw_data_test = missing_check(np.array(raw_data_test)[:, :-len(pollution_kind)])
Y_test = np.array(Y_test, dtype=np.float)

final_time = time.time()
print('Reading data .. ok, ', end='')
time_spent_printer(start_time, final_time)


# normalize
print('Normalize ..')
# mean_X_train = np.mean(X_train, axis=0)
# std_X_train = np.std(X_train, axis=0)
# if 0 in std_X_train:
#     input("Denominator can't be 0.")
# X_train = np.array([(x_train-mean_X_train)/std_X_train for x_train in X_train])
# X_test = np.array([(x_test-mean_X_train)/std_X_train for x_test in X_test])
#
mean_y_train = np.mean(Y_train)
std_y_train = np.std(Y_train)
if not std_y_train:
    input("Denominator can't be 0.")
Y_train = [(y - mean_y_train) / std_y_train for y in Y_train]
print('mean_y_train: %f  std_y_train: %f' % (mean_y_train, std_y_train))
#
# fw = open(folder + "%s_parameter_%s.pickle" % (target_site, target_kind), 'wb')
# cPickle.dump(str(mean_X_train) + ',' +
#              str(std_X_train) + ',' +
#              str(mean_y_train) + ',' +
#              str(std_y_train), fw)
# fw.close()


# feature process
print('feature process ..')
if 'WIND_DIREC' in pollution_kind:
    index_of_kind = pollution_kind.index('WIND_DIREC')
    length_of_kind_list = len(pollution_kind)
    len_of_sites_list = len(site_list)
    data_train = raw_data_train.tolist()
    data_test = raw_data_test.tolist()
    for i in range(len(data_train)):
        for j in range(len_of_sites_list):
            specific_index = feature_kind_shift + index_of_kind + j * length_of_kind_list
            coordin = data_coordinate_angle(data_train[i].pop(specific_index+j))  # *std_X_train[specific_index]+mean_X_train[specific_index]
            data_train[i].insert(specific_index + j, coordin[1])
            data_train[i].insert(specific_index + j, coordin[0])
            if i < len(data_test):
                coordin = data_coordinate_angle(data_test[i].pop(specific_index+j))  # *std_X_train[specific_index]+mean_X_train[specific_index]
                data_test[i].insert(specific_index + j, coordin[1])
                data_test[i].insert(specific_index + j, coordin[0])
    data_train = np.array(data_train)
    data_test = np.array(data_test)
else:
    data_train = raw_data_train
    data_test = raw_data_test

# --
del raw_data_train
del raw_data_test

# -- content of each group --
# pollution_kind = ['PM2.5', 'O3', 'SO2', 'CO', 'NO2', 'WIND_SPEED', 'WIND_DIREC']  # WIND_DIREC has two dimension
# pollution_group = list()
# append(['O3', 'SO2', 'WIND_SPEED', 'WIND_DIREC'])
# append(['O3', 'CO', 'WIND_SPEED', 'WIND_DIREC'])
# append(['O3', 'NO2', 'WIND_SPEED', 'WIND_DIREC'])
# append(['PM2.5', 'WIND_SPEED', 'WIND_DIREC'])
# X_train = list()
# for i in range(len(pollution_group)):
#     x_train = list()
#     for k in range(feature_kind_shift):
#         x_train.append(data_train[:, k])
#     for k in range(len(site_list)):
#         for j in range(len(pollution_group[i])):
#             index = pollution_kind.index(pollution_group[i][j]) + k * (len(pollution_kind)+1)
#             x_train.append(data_train[:, index])
#             if pollution_group[i][j] == 'WIND_DIREC':
#                 x_train.append(data_train[:, index+1])
#     X_train.append(np.array(x_train).T)
#
# X_test = list()
# for i in range(len(pollution_group)):
#     x_test = list()
#     for k in range(feature_kind_shift):
#         x_test.append(data_test[:, k])
#     for k in range(len(site_list)):
#         for j in range(len(pollution_group[i])):
#             index = pollution_kind.index(pollution_group[i][j]) + k * (len(pollution_kind)+1)
#             x_test.append(data_test[:, index])
#             if pollution_group[i][j] == 'WIND_DIREC':
#                 x_test.append(data_test[:, index+1])
#     X_test.append(np.array(x_test).T)
#
# del data_train
# del data_test

# --

X_train = data_train
X_test = data_test


print('Constructing time series data set ..')
start_time = time.time()
# for i in range(len(pollution_group)):
# for layer 1
X_train = construct_time_steps(X_train[:-1], layer1_time_steps)
X_test = construct_time_steps(X_test[:-1], layer1_time_steps)

# for layer 2
X_train = construct_second_time_steps(X_train, layer1_time_steps, layer2_time_steps)
X_test = construct_second_time_steps(X_test, layer1_time_steps, layer2_time_steps)

final_time = time.time()
time_spent_printer(start_time, final_time)


# for layer 1
Y_train = Y_train[layer1_time_steps:]
Y_test = Y_test[layer1_time_steps:]
# for layer 2
Y_train = Y_train[layer1_time_steps*(layer2_time_steps-1):]
Y_test = Y_test[layer1_time_steps*(layer2_time_steps-1):]

# --

Y_real = np.copy(Y_test)

# Y_train = higher(Y_train, interval_hours)
# Y_test = higher(Y_test, interval_hours)
Y_train = Y_train[interval_hours-1:]
Y_test = Y_test[interval_hours-1:]
Y_real = Y_real[interval_hours - 1:]

# --
min_length_X_train = len(X_train)
min_length_X_test = len(X_test)

# for i in range(len(pollution_group)):
#     try:
#         if len(X_train[i]) < min_length_X_train:
#             min_length_X_train = len(X_train[i])
#     except:
#         min_length_X_train = len(X_train[i])
#
#     try:
#         if len(X_test[i]) < min_length_X_test:
#             min_length_X_test = len(X_test[i])
#     except:
#         min_length_X_test = len(X_test[i])

train_seq_len = np.min([len(Y_train), min_length_X_train])
test_seq_len = np.min([len(Y_test), min_length_X_test])

print('%d train sequences' % train_seq_len)
print('%d test sequences' % test_seq_len)

# for i in range(len(pollution_group)):
X_train = X_train[:train_seq_len]
X_test = X_test[:test_seq_len]

Y_train = Y_train[:train_seq_len]
Y_test = Y_test[:test_seq_len]
Y_real = Y_real[:test_seq_len]


# -- fourier transfer --


def time_domain_to_frequency_domain(time_tensor):
    freq_output = list()
    for f_i in range(len(time_tensor)):
        freq_tensor = list()

        length_of_kind_list = len(pollution_kind)
        index_of_site = site_list.index(target_site)
        length_of_kind_list = length_of_kind_list + 1 if 'WIND_DIREC' in pollution_kind else length_of_kind_list

        for f_j in pollution_kind:
            if f_j == 'WIND_DIREC':
                continue

            freq_feature_matrix = list()

            index_of_kind = pollution_kind.index(f_j)
            index = feature_kind_shift + index_of_kind + index_of_site * length_of_kind_list

            for f_k in range(len(time_tensor[f_i])):
                freq_feature_matrix.append(np.real(fft(time_tensor[f_i, f_k, :, index])))

            freq_tensor.append(np.array(freq_feature_matrix))
        freq_output.append(freq_tensor)
    return freq_output


start_time = time.time()
print('fourier transfer .. ')
print('for training data ..')
freq_X_train = np.array(X_train)
freq_X_train = time_domain_to_frequency_domain(freq_X_train)
print('for testing data ..')
freq_X_test = np.array(X_test)
freq_X_test = time_domain_to_frequency_domain(freq_X_test)
final_time = time.time()
print('fourier transfer .. ok, ', end='')
time_spent_printer(start_time, final_time)

# --

# delete data which have missing values
i = 0
while i < len(Y_test):
    if not(Y_test[i] > -10000):  # check missing or not, if Y_test[i] is missing, then this command will return True
        Y_test = np.delete(Y_test, i, 0)
        Y_real = np.delete(Y_real, i, 0)
        # for i in range(len(pollution_group)):
        X_test = np.delete(X_test, i, 0)
        freq_X_test = np.delete(freq_X_test, i, 0)
        i = -1
    i += 1
Y_test = np.array(Y_test, dtype=np.float)
Y_real = np.array(Y_real, dtype=np.float)

print('delete invalid testing data, remain ', len(Y_test), 'test sequences')

# --

# for i in range(len(pollution_group)):
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)

freq_X_train = np.array(freq_X_train)
freq_X_test = np.array(freq_X_test)

# validation set
X_validation = X_train[-800:]
freq_X_validation = freq_X_train[-800:]
Y_validation = Y_train[-800:]

X_train = X_train[:-800]
freq_X_train = freq_X_train[:-800]
Y_train = Y_train[:-800]

print('take 800 data to validation set')

# np.random.seed(seed)
# np.random.shuffle(Y_train)
#
# np.random.seed(seed)
# np.random.shuffle(X_train)
"""
else:  # is_training = false
    # mean and std
    fr = open(folder + "%s_parameter_%s.pickle" % (target_site, target_kind), 'rb')
    [mean_X_train, std_X_train, mean_y_train, std_y_train] = (cPickle.load(fr)).split(',')
    mean_X_train = mean_X_train.replace('[', '').replace(']', '').replace('\n', '').split(' ')
    while '' in mean_X_train:
        mean_X_train.pop(mean_X_train.index(''))
    mean_X_train = np.array(mean_X_train, dtype=np.float)
    std_X_train = std_X_train.replace('[', '').replace(']', '').replace('\n', '').split(' ')
    while '' in std_X_train:
        std_X_train.pop(std_X_train.index(''))
    std_X_train = np.array(std_X_train, dtype=np.float)
    mean_y_train = float(mean_y_train)
    std_y_train = float(std_y_train)
    fr.close()

    # reading data
    print('preparing testing set ..')
    X_test = read_data_sets(sites=site_list + [target_site], date_range=np.atleast_1d(testing_year),
                            beginning=testing_duration[0], finish=testing_duration[-1],
                            feature_selection=pollution_kind, update=data_update)
    Y_test = np.array(X_test)[:, -len(pollution_kind):]
    Y_test = Y_test[:, pollution_kind.index(target_kind)]
    X_test = missing_check(np.array(X_test)[:, :-len(pollution_kind)])

    # normalize
    print('Normalize ..')
    if 0 in std_X_train:
        input("Denominator can't be 0.")
    X_test = np.array([(x_test-mean_X_train)/std_X_train for x_test in X_test])

    # feature process
    if 'WIND_DIREC' in pollution_kind:
        index_of_kind = pollution_kind.index('WIND_DIREC')
        length_of_kind_list = len(pollution_kind)
        len_of_sites_list = len(site_list)
        X_test = X_test.tolist()
        for i in range(len(X_test)):
            for j in range(len_of_sites_list):
                specific_index = index_of_kind + j * length_of_kind_list
                coordin = data_coordinate_angle(
                    (X_test[i].pop(specific_index + j)) * std_X_train[specific_index] + mean_X_train[
                        specific_index])
                X_test[i].insert(specific_index + j, coordin[1])
                X_test[i].insert(specific_index + j, coordin[0])
        X_test = np.array(X_test)
    Y_test = np.array(Y_test, dtype=np.float)

    # --

    print('Constructing time series data set ..')
    # for layer 1 -----------------------------------------------------------------------------------------------------------------
    X_test = construct_time_steps(X_test[:-1], layer1_time_steps)
    Y_test = Y_test[layer1_time_steps:]

    # for layer 2
    X_test = construct_second_time_steps(X_test, layer1_time_steps, layer2_time_steps)
    Y_test =Y_test[layer1_time_steps*(layer2_time_steps-1):]

    # --
    Y_real = np.copy(Y_test)

    # Y_test = higher(Y_test, interval_hours)
    Y_test = Y_test[interval_hours - 1:]
    Y_real = Y_real[interval_hours - 1:]

    test_seq_len = np.min([len(Y_test), len(X_test)])

    print(len(X_test), 'test sequences')

    X_test = X_test[:test_seq_len]

    Y_test = Y_test[:test_seq_len]
    Y_real = Y_real[:test_seq_len]

    # delete data which have missing values
    i = 0
    while i < len(Y_test):
        if not (Y_test[i] > -10000):  # check missing or not, if Y_test[i] is missing, then this command will return True
            Y_test = np.delete(Y_test, i, 0)
            Y_real = np.delete(Y_real, i, 0)
            X_test = np.delete(X_test, i, 0)
            i = -1
        i += 1
    Y_test = np.array(Y_test, dtype=np.float)

    print('delete invalid testing data, remain ', len(X_test), 'test sequences')

    # --

    X_test = np.array(X_test)
"""

# -- rnn --
print('- rnn -')

filename = ("rnn_%s_training_%s_m%s_to_%s_m%s_interval_%s_%s"
            % (target_site, training_year[0], training_begining, training_year[-1], training_deadline, interval_hours, target_kind))
print(filename)


# --

print('Build rnn model...')
start_time = time.time()

# input layer
# rnn
model_input = list()
# for i in range(len(pollution_group)):
input_size = len(pollution_kind)+1 if 'WIND_DIREC' in pollution_kind else len(pollution_kind)  # feature 'WIND_DIREC' has two dimension
input_size = input_size * len(site_list) + feature_kind_shift

input_shape = (layer1_time_steps, input_size)

for j in range(layer2_time_steps):
    model_input.append(Input(shape=input_shape, dtype='float32'))

# cnn
# model_input2 = list()
freq_high = layer2_time_steps
freq_width = layer1_time_steps
num_of_kind = len(pollution_kind)-1 if 'WIND_DIREC' in pollution_kind else len(pollution_kind)

if K.image_data_format() == 'channels_first':
    freq_X_train = freq_X_train.reshape(freq_X_train.shape[0], num_of_kind, 1, freq_high, freq_width)
    freq_X_validation = freq_X_validation.reshape(freq_X_validation.shape[0], num_of_kind, 1, freq_high, freq_width)
    freq_X_test = freq_X_test.reshape(freq_X_test.shape[0], num_of_kind, 1, freq_high, freq_width)
    freq_input_shape = (1, freq_high, freq_width)
else:
    freq_X_train = freq_X_train.reshape(freq_X_train.shape[0], num_of_kind, freq_high, freq_width, 1)
    freq_X_validation = freq_X_validation.reshape(freq_X_validation.shape[0], num_of_kind, freq_high, freq_width, 1)
    freq_X_test = freq_X_test.reshape(freq_X_test.shape[0], num_of_kind, freq_high, freq_width, 1)
    freq_input_shape = (freq_high, freq_width, 1)

for i in range(num_of_kind):
    model_input.append(Input(shape=freq_input_shape, dtype='float32'))


# layer 1
# rnn_model_layer2 = list()
# for j in range(len(pollution_group)):
rnn_model_layer1 = list()
for i in range(layer2_time_steps):
    rnn_model_layer1.append(BatchNormalization(beta_regularizer=None, epsilon=0.001, beta_initializer="zero",
                                               gamma_initializer="one", weights=None, gamma_regularizer=None,
                                               momentum=0.99, axis=-1)(model_input[i]))
    # return_sequences=True
    # recurrent_activation='relu', kernel_constraint=maxnorm(2.), recurrent_constraint=maxnorm(2.),
    # bias_constraint=maxnorm(2.)
    rnn_model_layer1[i] = LSTM(hidden_size1, kernel_regularizer=l2(regularizer),
                               recurrent_regularizer=l2(regularizer), bias_regularizer=l2(regularizer),
                               recurrent_dropout=recurrent_dropout)(rnn_model_layer1[i])
    rnn_model_layer1[i] = Dropout(dropout)(rnn_model_layer1[i])

# layer 2
rnn_model_layer2 = concatenate(rnn_model_layer1)

rnn_model_layer2 = Reshape((layer2_time_steps, hidden_size1))(rnn_model_layer2)

rnn_model_layer2 = BatchNormalization(beta_regularizer=None, epsilon=0.001, beta_initializer="zero",
                                         gamma_initializer="one", weights=None, gamma_regularizer=None,
                                         momentum=0.99, axis=-1)(rnn_model_layer2)
rnn_model_layer2 = LSTM(hidden_size2, kernel_regularizer=l2(regularizer), recurrent_regularizer=l2(regularizer),
                           bias_regularizer=l2(regularizer), recurrent_dropout=recurrent_dropout)(
    rnn_model_layer2)

rnn_model_layer2 = Dropout(dropout)(rnn_model_layer2)


# cnn
cnn_model_layer = list()
for i in range(num_of_kind):
    cnn_kernel_layer = list()
    for k in [freq_high/4, freq_high/2]:
        kernel_first_layer = Conv2D(kernel_num_1, kernel_size=(k, freq_width), activation='relu')(model_input[layer2_time_steps+i])
        # kernel_second_layer = Conv2D(kernel_num_2, kernel_size=(k, 1), activation='relu')(kernel_first_layer)
        cnn_kernel_layer.append(Flatten()(kernel_first_layer))

    cnn_model_layer.append(concatenate(cnn_kernel_layer))
    cnn_model_layer[i] = Dropout(dropout)(cnn_model_layer[i])

    cnn_model_layer[i] = BatchNormalization(beta_regularizer=None, epsilon=0.001, beta_initializer="zero",
                                            gamma_initializer="one", weights=None, gamma_regularizer=None,
                                            momentum=0.99, axis=-1)(cnn_model_layer[i])
    cnn_model_layer[i] = Dense(cnn_hidden_size1, activation='relu')(cnn_model_layer[i])
    cnn_model_layer[i] = Dropout(dropout)(cnn_model_layer[i])

cnn_model_layer = concatenate(cnn_model_layer)

# output layer
# if len(rnn_model_layer2) > 1:
#     output_layer = concatenate(rnn_model_layer2)
# else:
output_layer = concatenate([rnn_model_layer2, cnn_model_layer])


output_layer = BatchNormalization(beta_regularizer=None, epsilon=0.001, beta_initializer="zero", gamma_initializer="one",
                                  weights=None, gamma_regularizer=None, momentum=0.99, axis=-1)(output_layer)
output_layer = Dense(output_size, kernel_regularizer=l2(regularizer), bias_regularizer=l2(regularizer))(output_layer)


rnn_model = Model(inputs=model_input, outputs=output_layer)
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
    # X_train_input = list()
    # X_test_input = list()

    # for i in range(len(pollution_group)):
    #     X_train_input += ([X_train[i][:, j, :, :] for j in range(layer2_time_steps)])
    #     X_test_input += ([X_test[i][:, j, :, :] for j in range(layer2_time_steps)])
    X_train_input = [X_train[:, j, :, :] for j in range(layer2_time_steps)] + \
                    [freq_X_train[:, j, :, :] for j in range(num_of_kind)]
    X_validation_input = [X_validation[:, j, :, :] for j in range(layer2_time_steps)] + \
                         [freq_X_validation[:, j, :, :] for j in range(num_of_kind)]

    rnn_model.fit(X_train_input, Y_train,
                  batch_size=batch_size,
                  epochs=epoch,
                  validation_data=(X_validation_input, Y_validation),
                  # validation_split=0.05, validation_data=None,
                  shuffle=True,
                  callbacks=[
                      EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto'),
                      ModelCheckpoint(folder + filename, monitor='val_loss', verbose=0, save_best_only=False,
                                      save_weights_only=True, mode='auto', period=1)])

    # Potentially save weights
    rnn_model.save_weights(folder + filename, overwrite=True)
    # rnn_model.save(folder + filename, overwrite=True)

    final_time = time.time()
    time_spent_printer(start_time, final_time)
    print('model saved: ', filename)

else:
    print('loading model ..')
    # print('loading model from %s' % (folder + filename + ".hdf5"))
    # rnn_model.load_weights(folder + filename)
    # try:
    #     rnn_model = keras.models.load_model(folder + filename)
    # except:
    #     h5file_check(folder + filename)
    #     rnn_model = keras.models.load_model(folder + filename)
    rnn_model.load_weights(folder + filename)


# X_test_input = list()
# for i in range(len(pollution_group)):
#     X_test_input += ([X_test[i][:, j, :, :] for j in range(layer2_time_steps)])
X_test_input = [X_test[:, j, :, :] for j in range(layer2_time_steps)] + \
               [freq_X_test[:, j, :, :] for j in range(num_of_kind)]


rnn_pred = rnn_model.predict(X_test_input)
final_time = time.time()
time_spent_printer(start_time, final_time)

pred = mean_y_train + std_y_train * rnn_pred
# pred = rnn_pred

print('rmse(rnn): %.5f' % (np.mean((np.atleast_2d(Y_test).T - pred)**2, 0)**0.5))

# --


def plotting(data, filename, grid=[24, 10], save=False, show=False, collor=['mediumaquamarine', 'pink', 'gray']):
    if len(grid) != 2:
        print('len(grid) must equal to 2')
    for i in range(len(data)):
        c = i if i < len(collor) else i % len(collor)
        plt.plot(np.arange(len(data[i])), data[i], c=collor[c])

    plt.xticks(np.arange(0, len(data[0]), grid[0]))
    plt.yticks(np.arange(0, max(data[0]), grid[1]))
    plt.grid(True)
    plt.rc('axes', labelsize=4)
    if save:
        plt.savefig(root_path + 'result/' + filename)
    if show:
        plt.show()

plotting([Y_test, pred], filename + '.png', save=True, show=True)

# -- frequency --

Y_freq = np.real(fft(Y_real))
# with open(root_path + 'result/freq', 'w') as f:
#     # print(Y_freq)
#     for i in Y_freq:
#         f.write(str(i))
#         f.write(',')
plotting([Y_freq], 'freq.png', grid=[100, 1000], save=True, show=True)

# -- validation
X_validation_input = [X_validation[:, j, :, :] for j in range(layer2_time_steps)] + \
                     [freq_X_validation[:, j, :, :] for j in range(num_of_kind)]

valid_pred = rnn_model.predict(X_validation_input)
plotting([Y_validation, valid_pred], 'valid.png', save=True, show=True)

# -- check overfitting --

# X_train_input = list()
# for i in range(len(pollution_group)):
#     X_train_input += ([X_train[i][:, j, :, :][-800:] for j in range(layer2_time_steps)])
X_train_input = [X_train[:, j, :, :][-800:] for j in range(layer2_time_steps)] + \
                [freq_X_train[:, j, :, :][-800:] for j in range(num_of_kind)]

train_pred = rnn_model.predict(X_train_input)
train_pred_target = Y_train[-800:]

# train_pred = mean_y_train + std_y_train * train_pred
# train_pred_target = mean_y_train + std_y_train * train_pred_target

plotting([train_pred_target, train_pred], 'train.png', save=True, show=True)

