# _*_ coding: utf-8 _*_

# not longer used
# GPU command:
#     THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python script.py

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import time
import sys
import os
import cPickle
import matplotlib.pyplot as plt

# from keras.optimizers import SGD, RMSprop, Adagrad
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
# from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM  # , GRU, SimpleRNN
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization

from reader import read_data_sets, construct_time_steps
from missing_value_processer import missing_check
from feature_processor import data_coordinate_angle

root_path = '/home/clliao/workspace/python/weather_prediction/rnn_regression/'


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


# local = '北部'
# city = '台北'
# site_list = pollution_site_map[local][city]  # ['中山', '古亭', '士林', '松山', '萬華']
# target_site = '萬華'
#
# training_year = ['2014', '2016']  # change format from   2014-2015   to   ['2014', '2015']
# testing_year = ['2017', '2017']
#
# training_duration = ['1/1', '12/31']
# testing_duration = ['1/1', '1/31']
# interval_hours = 6  # predict the label of average data of many hours later, default is 1
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
pollution_kind = ['PM2.5', 'O3', 'AMB_TEMP', 'RH', 'WIND_SPEED', 'WIND_DIREC']
target_kind = 'PM2.5'
data_update = False
# batch_size = 24 * 7
seed = 0


# Network Parameters
input_size = (len(site_list)*len(pollution_kind)+len(site_list)) if 'WIND_DIREC' in pollution_kind else (len(site_list)*len(pollution_kind))
time_steps = 12
hidden_size = 20
output_size = 1

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

testing_month = testing_duration[0][:testing_duration[0].index('/')]
folder = root_path+"model/%s/%s/" % (local, city)
training_begining = training_duration[0][:training_duration[0].index('/')]
training_deadline = training_duration[-1][:training_duration[-1].index('/')]
filename = ("sa_DropoutLSTM_%s_training_%s_m%s_to_%s_m%s_interval_%s"
            % (target_site, training_year[0], training_begining, training_year[-1], training_deadline, interval_hours))
print(filename)
print('Training for %s/%s to %s/%s' % (training_year[0], training_duration[0], training_year[-1], training_duration[-1]))
print('Testing for %s/%s to %s/%s' % (testing_year[0], testing_duration[0], testing_year[-1], testing_duration[-1]))


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

fw = open(folder + filename + ".pickle", 'wb')
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
# Y_test = Y_test[time_steps:]
# --
print('Constructing time series data set ..')
X_train = construct_time_steps(X_train[:-1], time_steps)
# Y_train = construct_time_steps(Y_train[1:], time_steps)
Y_train = Y_train[time_steps:]

X_test = construct_time_steps(X_test[:-1], time_steps)
Y_test = Y_test[time_steps:]


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
            Y[i] = np.array(higher_list).sum()/reserve_hours
        if deadline:
            X = X[:deadline]
            Y = Y[:deadline]
    return X, Y

[X_train, Y_train] = higher(X_train, Y_train, interval_hours)
[X_test, Y_test] = higher(X_test, Y_test, interval_hours)


# delete data which have missing values
i = 0
while i < len(Y_test):
    if not(Y_test[i] > -10000):  # check missing or not, if Y_test[i] is missing, then this command will return True
        Y_test = np.delete(Y_test, i, 0)
        X_test = np.delete(X_test, i, 0)
        i = -1
    i += 1
Y_test = np.array(Y_test, dtype=np.float)
# --
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)

np.random.seed(seed)
np.random.shuffle(X_train)
np.random.seed(seed)
np.random.shuffle(Y_train)

# --

print('Build model...')
start_time = time.time()
model = Sequential()

# layer 1
model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
                             gamma_init='one', gamma_regularizer=None, beta_regularizer=None,
                             input_shape=(time_steps, input_size)))
model.add(LSTM(hidden_size, W_regularizer=l2(weight_decay), U_regularizer=l2(weight_decay),
               b_regularizer=l2(weight_decay), dropout_W=p_W, dropout_U=p_U))  # return_sequences=True
model.add(Dropout(p_dense))

# # layer 2
# model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
#                              gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
# model.add(LSTM(hidden_size, W_regularizer=l2(weight_decay), U_regularizer=l2(weight_decay),
#                b_regularizer=l2(weight_decay), dropout_W=p_W, dropout_U=p_U))
# model.add(Dropout(p_dense))

# output layer
model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
                             gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
model.add(Dense(output_size, W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay)))

# optimiser = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=False)
optimiser = 'adam'
model.compile(loss='mean_squared_error', optimizer=optimiser)

final_time = time.time()
time_spent_printer(start_time, final_time)

# --

if is_training:
    print("Train...")
    start_time = time.time()

    # modeltest_1 = ModelTest(X_train[:100],
    #                         mean_y_train + std_y_train * np.atleast_2d(Y_train[:100]).T,
    #                         test_every_X_epochs=1, verbose=0, loss='euclidean',
    #                         mean_y_train=mean_y_train, std_y_train=std_y_train)
    # modeltest_2 = ModelTest(X_test, np.atleast_2d(Y_test).T, test_every_X_epochs=1,
    #                         verbose=0, loss='euclidean',
    #                         mean_y_train=mean_y_train, std_y_train=std_y_train)
    # model.fit(X_train, Y_train, batch_size=batch_size, epochs=5,
    #           callbacks=[modeltest_1, modeltest_2])
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=50)

    # Potentially save weights
    # model.save_weights("path", overwrite=True)

    final_time = time.time()
    time_spent_printer(start_time, final_time)

else:
    print('loading model ..')
    # print('loading model from %s' % (folder + filename + ".hdf5"))
    model.load_weights(folder + filename + ".hdf5")

# --

print("Test...")

# Evaluate model
# Dropout approximation for training data:
standard_prob = model.predict(X_train, batch_size=500, verbose=1)
print(np.mean(((mean_y_train + std_y_train * np.atleast_2d(Y_train).T)
               - (mean_y_train + std_y_train * standard_prob))**2, 0)**0.5)

# --

# Dropout approximation for test data:
standard_prob = model.predict(X_test, batch_size=500, verbose=1)
print(np.mean((np.atleast_2d(Y_test).T - (mean_y_train + std_y_train * standard_prob))**2, 0)**0.5)

# MC dropout for test data:
# T = 50
# prob = np.array([modeltest_2.predict_stochastic(X_test, batch_size=500, verbose=0)
#                  for _ in xrange(T)])
# rnn_pred = np.mean(prob, 0)

rnn_pred = standard_prob

print(np.mean((np.atleast_2d(Y_test).T - (mean_y_train + std_y_train * rnn_pred))**2, 0)**0.5)


# --

def target_level(target, kind='PM2.5'):
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

# standard_prob_pred = np.zeros(len(standard_prob))
rnn_pred_value = np.zeros(len(rnn_pred))
rnn_pred_label = np.zeros(len(rnn_pred))
real_target = np.zeros(len(Y_test))

# standard_prob_true = 0.
# standard_prob_false = 0.
rnn_pred_true = 0.
rnn_pred_false = 0.

four_label_true = 0.0
four_label_false = 0.0

# calculate the accuracy of ten level
for i in range(len(rnn_pred)):
    # standard_prob_pred[i] = target_level(mean_y_train + std_y_train * rnn_pred[i])
    rnn_pred_value[i] = mean_y_train + std_y_train * rnn_pred[i]
    rnn_pred_label[i] = target_level(rnn_pred_value[i])
    real_target[i] = target_level(Y_test[i])

    # if real_target[i] == standard_prob_pred[i]:
    #     standard_prob_true += 1
    # else:
    #     standard_prob_false += 1

    if real_target[i] == rnn_pred_label[i]:
        rnn_pred_true += 1
    else:
        rnn_pred_false += 1

    # four label
    if (real_target[i] >= 1 and real_target[i] <= 3) and (rnn_pred_label[i] >= 1 and rnn_pred_label[i] <= 3):
        four_label_true += 1
    elif (real_target[i] >= 4 and real_target[i] <= 6) and (rnn_pred_label[i] >= 4 and rnn_pred_label[i] <= 6):
        four_label_true += 1
    elif (real_target[i] >= 7 and real_target[i] <= 9) and (rnn_pred_label[i] >= 7 and rnn_pred_label[i] <= 9):
        four_label_true += 1
    elif (real_target[i] >= 10) and (rnn_pred_label[i] >= 10):
        four_label_true += 1
    else:
        four_label_false += 1

# print('standard_prob_accuracy: %.5f' % (standard_prob_true / (standard_prob_true + standard_prob_false)))
print('rnn_pred_accuracy: %.5f' % (rnn_pred_true / (rnn_pred_true + rnn_pred_false)))
print('Four level accuracy: %.5f' % (four_label_true / (four_label_true + four_label_false)))
print('--')

ha = 0.0  # observation high, predict high
hb = 0.0  # observation low, predict high
hc = 0.0  # observation high, predict low
hd = 0.0  # observation low, predict low
la = 0.0  # observation very high, predict very high
lb = 0.0
lc = 0.0
ld = 0.0

# two_label_true = 0.0
# two_label_false = 0.0
# statistic of status of prediction by label of forecast & observation
# for each_label in np.arange(len(real_target)):
#     if real_target[each_label] >= 7:  # observation high
#         if rnn_pred_label[each_label] >= 7:
#             ha += 1
#             # two_label_true += 1
#         else:
#             hc += 1
#             # two_label_false += 1
#     else:  # observation low
#         if rnn_pred_label[each_label] >= 7:
#             hb += 1
#             # two_label_false += 1
#         else:
#             hd += 1
#             # two_label_true += 1
#
#     if real_target[each_label] >= 4:  # observation higher
#         if rnn_pred_label[each_label] >= 4:
#             la += 1
#         else:
#             lc += 1
#     else:  # observation very low
#         if rnn_pred_label[each_label] >= 4:
#             lb += 1
#         else:
#             ld += 1


# statistic of status of prediction by value of forecast & observation
for each_value in range(len(Y_test)):
    if Y_test[each_value] >= 53.5:  # observation high
        if rnn_pred_value[each_value] >= 53.5:  # forecast high(with tolerance)
            ha += 1
        else:
            hc += 1
    else:  # observation low
        if rnn_pred_value[each_value] >= 53.5:
            hb += 1
        else:
            hd += 1

    if Y_test[each_value] >= 35.5:  # observation higher
        if rnn_pred_value[each_value] >= 35.5:
            la += 1
        else:
            lc += 1
    else:  # observation very low
        if rnn_pred_value[each_value] >= 35.5:
            lb += 1
        else:
            ld += 1

# print('Two level accuracy: %f' % (two_label_true / (two_label_true + two_label_false)))
print('high label: (%d, %d, %d, %d)' % (ha, hb, hc, hd))
print('low label: (%d, %d, %d, %d)' % (la, lb, lc, ld))

# plot the real trend and trend of prediction
prediction = mean_y_train + std_y_train * rnn_pred
plt.plot(np.arange(len(prediction)), Y_test[:len(prediction)], c='gray')
plt.plot(np.arange(len(prediction)), prediction, color='pink')

plt.xticks(np.arange(0, len(prediction), 24))
plt.yticks(np.arange(0, max(Y_test), 10))
plt.grid(True)
plt.rc('axes', labelsize=4)

if True:  # is_training:
    print('Writing result ..')
    with open(root_path + 'result/%s/%s/%s_training_%s_m%s_to_%s_m%s_testing_%s_m%s_ave%d.ods' % (local, city, target_site, training_year[0], training_begining, training_year[-1], training_deadline, testing_year[0], testing_month, interval_hours), 'wt') as f:
        print('RMSE: %f' % (np.sqrt(np.mean((np.atleast_2d(Y_test).T - (mean_y_train + std_y_train * rnn_pred))**2))), file=f)
        f.write('\n')
        print('Ten level accuracy: %f' % (rnn_pred_true / (rnn_pred_true + rnn_pred_false)), file=f)
        f.write('\n')
        print('Four level accuracy: %f' % (four_label_true / (four_label_true + four_label_false)), file=f)
        f.write('\n')
        # print('Two level accuracy: %f' % (two_label_true / (two_label_true + two_label_false)), file=f)
        # f.write('\n')
        print('high label:, %d, %d, %d, %d' % (ha, hb, hc, hd), file=f)
        f.write('\n')
        print('low label:, %d, %d, %d, %d' % (la, lb, lc, ld), file=f)
        f.write('\n')
        try:
            print('precision:, %f' % (ha / (ha + hb)), file=f)
        except:
            print('precision:, -1', file=f)
        f.write('\n')
        try:
            print('recall:, %f' % (ha / (ha + hc)), file=f)
        except:
            print('recall:, -1', file=f)
        f.write('\n')
        try:
            print('f1 score:, %f' % ((2 * (ha / (ha + hb)) * (ha / (ha + hc))) / ((ha / (ha + hb)) + (ha / (ha + hc)))),
                  file=f)
        except:
            print('f1 score:, -1', file=f)
        f.write('\n')
    print('Writing result .. ok')
    plt.savefig(root_path + 'result/%s/%s/%s_training_%s_m%s_to_%s_m%s_testing_%s_m%s_ave%d.png' % (local, city, target_site, training_year[0], training_begining, training_year[-1], training_deadline, testing_year[0], testing_month, interval_hours), dpi=100)
else:
    plt.show()
