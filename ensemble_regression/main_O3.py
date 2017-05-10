# _*_ coding: utf-8 _*_

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import time
import cPickle
import os

import xgboost as xgb
import matplotlib.pyplot as plt

# from keras.optimizers import SGD, RMSprop, Adagrad
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
# from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM  # , GRU, SimpleRNN
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization

from reader import read_data_sets, concatenate_time_steps, construct_time_steps
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


high_alert = 150.5
low_alert = 100.5

# local = '北部'  # 高屏 北部 中部
# city = '台北'  # 高雄 台北 台中
# site_list = pollution_site_map[local][city]  # ['中山', '古亭', '士林', '松山', '萬華']
# target_site = '萬華'  # 左營 中山 小港 大里
#
# training_year = ['2014', '2016']  # change format from   2014-2015   to   ['2014', '2015']
# testing_year = ['2017', '2017']
#
# training_duration = ['1/1', '12/31']
# testing_duration = ['1/1', '1/31']
# interval_hours = 8  # predict the label of average data of many hours later, default is 1
# is_training = True

local = os.sys.argv[1]
city = os.sys.argv[2]
site_list = pollution_site_map[local][city]
target_site = os.sys.argv[3]

training_year = [os.sys.argv[4][:os.sys.argv[4].index('-')], os.sys.argv[4][os.sys.argv[4].index('-')+1:]]  # change format from   2014-2015   to   ['2014', '2015']
testing_year = [os.sys.argv[5][:os.sys.argv[5].index('-')], os.sys.argv[5][os.sys.argv[5].index('-')+1:]]

training_duration = [os.sys.argv[6][:os.sys.argv[6].index('-')], os.sys.argv[6][os.sys.argv[6].index('-')+1:]]
testing_duration = [os.sys.argv[7][:os.sys.argv[7].index('-')], os.sys.argv[7][os.sys.argv[7].index('-')+1:]]
interval_hours = int(os.sys.argv[8])  # predict the label of average data of many hours later, default is 1
is_training = True if (os.sys.argv[9] == 'True' or os.sys.argv[9] == 'true') else False  # True False

# target_kind = 'PM2.5'
target_kind = 'O3'  # 8 hours
pred_type = 'higher'  # ave higher

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
if target_kind == 'PM2.5':
    pollution_kind = ['PM2.5', 'O3', 'AMB_TEMP', 'RH', 'WIND_SPEED', 'WIND_DIREC']
elif target_kind == 'O3':
    pollution_kind = ['O3', 'NOx', 'AMB_TEMP', 'RH', 'WIND_SPEED', 'WIND_DIREC']
data_update = False
# batch_size = 24 * 7
seed = 0


# Network Parameters
input_size = (len(site_list)*len(pollution_kind)+len(site_list)) if 'WIND_DIREC' in pollution_kind else (len(site_list)*len(pollution_kind))
time_steps = 12
# hidden_size = 20
output_size = 1

testing_month = testing_duration[0][:testing_duration[0].index('/')]
folder = root_path+"model/%s/%s/%sh/%s/" % (local, city, interval_hours, target_kind)
if target_kind == 'O3':
    folder = root_path + "model/%s/%s/%sh/%s/%s/" % (local, city, interval_hours, target_kind, pred_type)
training_begining = training_duration[0][:training_duration[0].index('/')]
training_deadline = training_duration[-1][:training_duration[-1].index('/')]
print('site: %s' % target_site)
print('Training for %s/%s to %s/%s' % (training_year[0], training_duration[0], training_year[-1], training_duration[-1]))
print('Testing for %s/%s to %s/%s' % (testing_year[0], testing_duration[0], testing_year[-1], testing_duration[-1]))
print('Target: %s' % target_kind)
if target_kind == 'O3':
    print('type: %s' % pred_type)


# for interval
def ave(X, Y, interval_hours):
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
        X = X[:deadline]
        Y = Y[:deadline]
    return X, Y


# for interval
def higher(X, Y, interval_hours):
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
            X = X[:deadline]
            Y = Y[:deadline]
    return X, Y

if is_training:
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

    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    if (len(X_train) < time_steps) or (len(X_test) < time_steps):
        input('time_steps(%d) too long.' % time_steps)

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

    fw = open(folder + "%s_parameter.pickle" % target_site, 'wb')
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
                X_train[i].insert(specific_index, coordin[1])
                X_train[i].insert(specific_index, coordin[0])
                if i < len(X_test):
                    coordin = data_coordinate_angle((X_test[i].pop(specific_index+j))*std_X_train[specific_index]+mean_X_train[specific_index])
                    X_test[i].insert(specific_index, coordin[1])
                    X_test[i].insert(specific_index, coordin[0])
        X_train = np.array(X_train)
        X_test = np.array(X_test)
    Y_test = np.array(Y_test, dtype=np.float)

    # --

    print('Constructing time series data set ..')
    # for rnn
    X_rnn_train = construct_time_steps(X_train[:-1], time_steps)
    X_rnn_test = construct_time_steps(X_test[:-1], time_steps)

    X_train = concatenate_time_steps(X_train[:-1], time_steps)
    Y_train = Y_train[time_steps:]

    X_test = concatenate_time_steps(X_test[:-1], time_steps)
    Y_test = Y_test[time_steps:]

    if target_kind == 'PM2.5':
        [X_train, Y_train] = higher(X_train, Y_train, interval_hours)
        [X_test, Y_test] = higher(X_test, Y_test, interval_hours)
    elif target_kind == 'O3':
        if pred_type == 'ave':
            [X_train, Y_train] = ave(X_train, Y_train, interval_hours)
            [X_test, Y_test] = ave(X_test, Y_test, interval_hours)
        if pred_type == 'higher':
            [X_train, Y_train] = higher(X_train, Y_train, interval_hours)
            [X_test, Y_test] = higher(X_test, Y_test, interval_hours)

    X_rnn_train = X_rnn_train[:len(X_train)]
    X_rnn_test = X_rnn_test[:len(X_test)]

    # delete data which have missing values
    i = 0
    while i < len(Y_test):
        if not(Y_test[i] > -10000):  # check missing or not, if Y_test[i] is missing, then this command will return True
            Y_test = np.delete(Y_test, i, 0)
            X_test = np.delete(X_test, i, 0)
            X_rnn_test = np.delete(X_rnn_test, i, 0)
            i = -1
        i += 1
    Y_test = np.array(Y_test, dtype=np.float)

    # --

    X_rnn_train = np.array(X_rnn_train)
    X_rnn_test = np.array(X_rnn_test)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)

    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(Y_train)

    np.random.seed(seed)
    np.random.shuffle(X_rnn_train)

else:  # is_training = false
    # mean and std
    fr = open(folder + "%s_parameter.pickle" % target_site, 'rb')
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
                X_test[i].insert(specific_index, coordin[1])
                X_test[i].insert(specific_index, coordin[0])
        X_test = np.array(X_test)
    Y_test = np.array(Y_test, dtype=np.float)

    # --

    print('Constructing time series data set ..')
    X_rnn_test = construct_time_steps(X_test[:-1], time_steps)
    X_test = concatenate_time_steps(X_test[:-1], time_steps)
    Y_test = Y_test[time_steps:]

    if target_kind == 'PM2.5':
        [X_test, Y_test] = higher(X_test, Y_test, interval_hours)
    elif target_kind == 'O3':
        if pred_type == 'ave':
            [X_test, Y_test] = ave(X_test, Y_test, interval_hours)
        if pred_type == 'higher':
            [X_test, Y_test] = higher(X_test, Y_test, interval_hours)

    X_rnn_test = X_rnn_test[:len(X_test)]

    # delete data which have missing values
    i = 0
    while i < len(Y_test):
        if not (Y_test[i] > -10000):  # check missing or not, if Y_test[i] is missing, then this command will return True
            Y_test = np.delete(Y_test, i, 0)
            X_test = np.delete(X_test, i, 0)
            X_rnn_test = np.delete(X_rnn_test, i, 0)
            i = -1
        i += 1
    Y_test = np.array(Y_test, dtype=np.float)

    # --

    X_rnn_test = np.array(X_rnn_test)
    X_test = np.array(X_test)

# -- xgboost --
print('- xgboost -')

filename = ("xgboost_%s_training_%s_m%s_to_%s_m%s_interval_%s"
            % (target_site, training_year[0], training_begining, training_year[-1], training_deadline, interval_hours))
print(filename)

if is_training:
    xgb_model = xgb.XGBRegressor().fit(X_train, Y_train)

    fw = open(folder + filename, 'wb')
    cPickle.dump(xgb_model, fw)
    fw.close()
else:
    fr = open(folder + filename, 'rb')
    xgb_model = cPickle.load(fr)
    fr.close()

xgb_pred = xgb_model.predict(X_test)

print('rmse(xgboost): %.5f' % (np.mean((Y_test - (mean_y_train + std_y_train * xgb_pred))**2, 0)**0.5))

# -- random forest --
# print('- random forest -')
#
# filename = ("random_forest_%s_training_%s_m%s_to_%s_m%s_interval_%s"
#             % (target_site, training_year[0], training_begining, training_year[-1], training_deadline, interval_hours))
# print(filename)
#
# rf_model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
#                                  max_features='auto', max_leaf_nodes=None, min_samples_leaf=2,
#                                  min_samples_split=10, min_weight_fraction_leaf=0.0,
#                                  n_estimators=10, n_jobs=5, oob_score=False, random_state=None,
#                                  verbose=0, warm_start=False)
# rf_model.fit(X_train, Y_train)
# rf_pred = rf_model.predict(X_test)
#
# print('rmse(random forest): %.5f' % (np.mean((Y_test - (mean_y_train + std_y_train * rf_pred))**2, 0)**0.5))

# -- rnn --
print('- rnn -')

filename = ("sa_DropoutLSTM_%s_training_%s_m%s_to_%s_m%s_interval_%s"
            % (target_site, training_year[0], training_begining, training_year[-1], training_deadline, interval_hours))
print(filename)

# Network Parameters
time_steps = 12
hidden_size = 20

# if len(sys.argv) == 1:
#     print("Expected args: p_W, p_U, p_dense, p_emb, weight_decay, batch_size, maxlen")
#     print("Using default args:")
#     sys.argv = ["", "0.5", "0.5", "0.5", "0.5", "1e-6", "128", "200"]
print("Expected args: p_W, p_U, p_dense, p_emb, weight_decay, batch_size")
print("Using default args:")
param = ["", "0.5", "0.5", "0.5", "0.5", "1e-6", "128"]
# sys.argv = ["", "0.25", "0.25", "0.25", "0.25", "1e-4", "128"]
# args = [float(a) for a in sys.argv[1:]]
args = [float(a) for a in param[1:]]
print(args)
p_W, p_U, p_dense, p_emb, weight_decay, batch_size = args
batch_size = int(batch_size)

# --

print('Build rnn model...')
start_time = time.time()
rnn_model = Sequential()

# layer 1
rnn_model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
                                 gamma_init='one', gamma_regularizer=None, beta_regularizer=None,
                                 input_shape=(time_steps, input_size)))
rnn_model.add(LSTM(hidden_size, W_regularizer=l2(weight_decay), U_regularizer=l2(weight_decay),
                   b_regularizer=l2(weight_decay), dropout_W=p_W, dropout_U=p_U))  # return_sequences=True  # recurrent_dropout=0.0
rnn_model.add(Dropout(p_dense))

# layer 2
# rnn_model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
#                                  gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
# rnn_model.add(LSTM(hidden_size, W_regularizer=l2(weight_decay), U_regularizer=l2(weight_decay),
#                    b_regularizer=l2(weight_decay), dropout_W=p_W, dropout_U=p_U))
# rnn_model.add(Dropout(p_dense))

# output layer
rnn_model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
                                 gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
rnn_model.add(Dense(output_size, W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay)))

# optimiser = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=False)
optimiser = 'adam'
rnn_model.compile(loss='mean_squared_error', optimizer=optimiser)

final_time = time.time()
time_spent_printer(start_time, final_time)


if is_training:
    print("Train...")
    start_time = time.time()
    rnn_model.fit(X_rnn_train, Y_train, batch_size=batch_size, epochs=50)

    # Potentially save weights
    rnn_model.save_weights(folder + filename, overwrite=True)

    final_time = time.time()
    time_spent_printer(start_time, final_time)

else:
    print('loading model ..')
    # print('loading model from %s' % (folder + filename + ".hdf5"))
    rnn_model.load_weights(folder + filename)

rnn_pred = rnn_model.predict(X_rnn_test, batch_size=500, verbose=1)
final_time = time.time()
time_spent_printer(start_time, final_time)
print('rmse(rnn): %.5f' % (np.mean((np.atleast_2d(Y_test).T - (mean_y_train + std_y_train * rnn_pred))**2, 0)**0.5))

# --  ensemble --

print('stacking ..')
if is_training:
    xgb_output = xgb_model.predict(X_train).reshape(len(X_train), 1)
    # rf_output = rf_model.predict(X_train).reshape(len(X_train), 1)
    rnn_output = rnn_model.predict(X_rnn_train, batch_size=500, verbose=1)
    # ensemble_X_train = np.hstack((X_train, xgb_output, rf_output, rnn_output))
    ensemble_X_train = np.hstack((X_train, xgb_output, rnn_output))

    Y_alert_train = [y * std_y_train + mean_y_train for y in Y_train]
    for element in range(len(Y_train)):
        if Y_alert_train[element] > high_alert:
            Y_alert_train[element] = 1  # [1, 0] = [high, low]
        else:
            Y_alert_train[element] = 0


xgb_pred = xgb_pred.reshape(len(X_test), 1)
# rf_pred = rf_pred.reshape(len(X_test), 1)
rnn_pred = rnn_pred.reshape(len(X_test), 1)
# ensemble_X_test = np.hstack((X_test, xgb_pred, rf_pred, rnn_pred))
ensemble_X_test = np.hstack((X_test, xgb_pred, rnn_pred))

Y_alert_test = np.zeros(len(Y_test))
for element in range(len(Y_test)):
    if Y_test[element] > high_alert:
        Y_alert_test[element] = 1  # [1, 0] = [high, low]

print('\n- ensemble -')
filename = ("ensemble_%s_training_%s_m%s_to_%s_m%s_interval_%s"
            % (target_site, training_year[0], training_begining, training_year[-1], training_deadline, interval_hours))
filename2 = ("classification_%s_training_%s_m%s_to_%s_m%s_interval_%s"
             % (target_site, training_year[0], training_begining, training_year[-1], training_deadline, interval_hours))

if is_training:
    ensemble_model = xgb.XGBRegressor().fit(ensemble_X_train, Y_train)
    classification_model = xgb.XGBClassifier().fit(ensemble_X_train, Y_alert_train)

    fw = open(folder + filename, 'wb')
    cPickle.dump(ensemble_model, fw)
    fw.close()

    fw2 = open(folder + filename2, 'wb')
    cPickle.dump(classification_model, fw2)
    fw2.close()
else:
    fr = open(folder + filename, 'rb')
    ensemble_model = cPickle.load(fr)
    fr.close()

    fr2 = open(folder + filename2, 'rb')
    classification_model = cPickle.load(fr2)
    fr2.close()

pred = ensemble_model.predict(ensemble_X_test)
alert_pred = classification_model.predict(ensemble_X_test)

# --

predictions = mean_y_train + std_y_train * pred
# print('mse: %.5f' % mean_squared_error(Y_test, predictions))
print('rmse: %.5f' % (np.mean((Y_test - predictions)**2, 0)**0.5))


def target_level(target, kind='O3'):
    # target should be a 1d-list
    if kind == 'PM2.5':
        if (target >= 0) and (target < 11.5):                # 0-11
            return 1
        elif (target >= 11.5) and (target < 23.5):           # 12-23
            return 2
        elif (target >= 23.5) and (target < 35.5):           # 24-35
            return 3
        elif (target >= 35.5) and (target < 41.5):           # 36-41
            return 4
        elif (target >= 41.5) and (target < 47.5):           # 42-47
            return 5
        elif (target >= 47.5) and (target < 53.5):           # 48-53
            return 6
        elif (target >= 53.5) and (target < 58.5):           # 54-58
            return 7
        elif (target >= 58.5) and (target < 64.5):           # 59-64
            return 8
        elif (target >= 64.5) and (target < 70.5):           # 65-70
            return 9
        elif target >= 70.5:                                                # others(71+)
            return 10
        else:
            print('error value: %d' % target)
            return 1
    elif kind == 'O3':
        if (target >= 0) and (target < 54.5):                # AQI 0~50
            return 1
        elif (target >= 54.5) and (target < 70.5):           # AQI 51~100
            return 2
        elif (target >= 70.5) and (target < 85.5):           # AQI 101~150
            return 3
        elif (target >= 85.5) and (target < 105.5):           # AQI 151~200
            return 4
        elif (target >= 105.5) and (target < 200.5):           # AQI 201~300
            return 5
        elif target >= 200.5:           # AQI 300
            return 5
        else:
            print('error value: %d' % target)
            return 1

pred_label = np.zeros(len(predictions))
real_target = np.zeros(len(Y_test))

pred_label_true = 0.
pred_label_false = 0.

# four_label_true = 0.0
# four_label_false = 0.0

# calculate the accuracy of AQI index
for i in range(len(predictions)):
    pred_label[i] = target_level(predictions[i])
    real_target[i] = target_level(Y_test[i])

    if real_target[i] == pred_label[i]:
        pred_label_true += 1
    else:
        pred_label_false += 1


# print('standard_prob_accuracy: %.5f' % (standard_prob_true / (standard_prob_true + standard_prob_false)))
print('AQI level accuracy: %.5f' % (pred_label_true / (pred_label_true + pred_label_false)))
print('--')

# --

ha = 0.0  # observation high, predict high
hb = 0.0  # observation low, predict high
hc = 0.0  # observation high, predict low
hd = 0.0  # observation low, predict low
la = 0.0  # observation very high, predict very high
lb = 0.0
lc = 0.0
ld = 0.0
alert_a = 0.0
alert_b = 0.0
alert_c = 0.0
alert_d = 0.0
integration_a = 0.0
integration_b = 0.0
integration_c = 0.0
integration_d = 0.0

for each_value in range(len(Y_test)):
    if Y_test[each_value] >= high_alert:  # observation high
        # regression
        if predictions[each_value] >= high_alert:  # forecast high(with tolerance)
            ha += 1
        else:
            hc += 1

        # classification
        if alert_pred[each_value]:  # [1, 0] = [high, low]
            alert_a += 1
        else:
            alert_c += 1

        # integration
        if alert_pred[each_value] or (predictions[each_value] >= high_alert):
            integration_a += 1
        else:
            integration_c += 1

    else:  # observation low
        # regression
        if predictions[each_value] >= high_alert:
            hb += 1
        else:
            hd += 1

        # classification
        if alert_pred[each_value]:
            alert_b += 1
        else:
            alert_d += 1

        # integration
        if alert_pred[each_value] or (predictions[each_value] >= high_alert):
            integration_b += 1
        else:
            integration_d += 1

    # --------------------------------------------------------

    if Y_test[each_value] >= low_alert:  # observation higher
        if predictions[each_value] >= low_alert:
            la += 1
        else:
            lc += 1
    else:  # observation very low
        if predictions[each_value] >= low_alert:
            lb += 1
        else:
            ld += 1

print('site: %s interval: %d' % (target_site, interval_hours))
print('Training for %s/%s to %s/%s' % (training_year[0], training_duration[0], training_year[-1], training_duration[-1]))
print('Testing for %s/%s to %s/%s' % (testing_year[0], testing_duration[0], testing_year[-1], testing_duration[-1]))
print('Target: %s' % target_kind)
print('Feature: ', pollution_kind)

print('rmse: %.5f' % (np.mean((Y_test - predictions)**2, 0)**0.5))
print('AQI level accuracy: %.5f' % (pred_label_true / (pred_label_true + pred_label_false)))

# print('Two level accuracy: %f' % (two_label_true / (two_label_true + two_label_false)))
print('high label: (%d, %d, %d, %d)' % (ha, hb, hc, hd))
print('low label: (%d, %d, %d, %d)' % (la, lb, lc, ld))
print('alert: (%d, %d, %d, %d)' % (alert_a, alert_b, alert_c, alert_d))

plt.plot(np.arange(len(predictions)), Y_test[:len(predictions)], c='gray')
plt.plot(np.arange(len(predictions)), predictions, color='pink')
plt.xticks(np.arange(0, len(predictions), 24))
plt.yticks(np.arange(0, max(Y_test), 10))
plt.grid(True)
plt.rc('axes', labelsize=4)
# set('fontsize', 30)
# plt.show()


if True:  # is_training:
    print('Writing result ..')
    folder = 'result/%s/%s/%s/' % (local, city, target_kind)
    if target_kind == 'O3':
        folder = 'result/%s/%s/%s/%s/' % (local, city, target_kind, pred_type)
    with open(root_path + folder + '%s_training_%s_m%s_to_%s_m%s_testing_%s_m%s_ave%d.ods' % (target_site, training_year[0], training_begining, training_year[-1], training_deadline, testing_year[0], testing_month, interval_hours), 'wt') as f:
        print('RMSE: %f' % (np.sqrt(np.mean((Y_test - predictions)**2))), file=f)
        f.write('\n')
        print('AQI level accuracy: %f' % (pred_label_true / (pred_label_true + pred_label_false)), file=f)
        f.write('\n')
        # print('Four level accuracy: %f' % (four_label_true / (four_label_true + four_label_false)), file=f)
        # f.write('\n')
        print('alert_classification:, %d, %d, %d, %d' % (alert_a, alert_b, alert_c, alert_d), file=f)
        f.write('\n')
        # print('Two level accuracy: %f' % (two_label_true / (two_label_true + two_label_false)), file=f)
        # f.write('\n')
        print('alert_regression:, %d, %d, %d, %d' % (ha, hb, hc, hd), file=f)
        f.write('\n')
        print('alert_interation:, %d, %d, %d, %d' % (integration_a, integration_b, integration_c, integration_d), file=f)
        f.write('\n')
        print('low label:, %d, %d, %d, %d' % (la, lb, lc, ld), file=f)
        f.write('\n')
        try:
            print('precision:, %f' % (integration_a / (integration_a + integration_b)), file=f)
        except:
            print('precision:, -1', file=f)
        f.write('\n')
        try:
            print('recall:, %f' % (integration_a / (integration_a + integration_c)), file=f)
        except:
            print('recall:, -1', file=f)
        f.write('\n')
        try:
            print('f1 score:, %f' % ((2 * (integration_a / (integration_a + integration_b)) * (integration_a / (integration_a + integration_c))) / ((integration_a / (integration_a + integration_b)) + (integration_a / (integration_a + integration_c)))),
                  file=f)
        except:
            print('f1 score:, -1', file=f)
        f.write('\n')
    print('Writing result .. ok')
    plt.savefig(root_path + folder + '%s_training_%s_m%s_to_%s_m%s_testing_%s_m%s_ave%d.png' % (target_site, training_year[0], training_begining, training_year[-1], training_deadline, testing_year[0], testing_month, interval_hours), dpi=100)
else:
    plt.show()
