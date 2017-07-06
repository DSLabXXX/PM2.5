# _*_ coding: utf-8 _*_

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import time
import cPickle
import matplotlib.pyplot as plt

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


local = '北部'
city = '台北'
site_list = pollution_site_map[local][city]  # ['中山', '古亭', '士林', '松山', '萬華']
target_site = '萬華'

training_year = ['2016', '2016']  # change format from   2014-2015   to   ['2014', '2015']
testing_year = ['2016', '2016']

training_duration = ['1/1', '10/31']
testing_duration = ['12/1', '12/31']
interval_hours = 24  # predict the label of average data of many hours later, default is 1
is_training = True


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
pollution_kind = ['PM2.5']  # , 'O3', 'SO2', 'CO', 'NO2', 'WIND_SPEED', 'WIND_DIREC', 'AMB_TEMP', 'RH'
data_update = False


testing_month = testing_duration[0][:testing_duration[0].index('/')]
folder = root_path+"model/%s/%s/%sh/" % (local, city, interval_hours)
training_begining = training_duration[0][:training_duration[0].index('/')]
training_deadline = training_duration[-1][:training_duration[-1].index('/')]
print('site: %s' % target_site)
print('Training for %s/%s to %s/%s' % (training_year[0], training_duration[0], training_year[-1], training_duration[-1]))
print('Testing for %s/%s to %s/%s' % (testing_year[0], testing_duration[0], testing_year[-1], testing_duration[-1]))
print('Target: %s' % target_kind)
# -------------------------------


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

# -- data collector

cd = 24
oclock = 8
collector = list()
counter = cd

for i in range(len(Y_test)):
    if counter == cd-8:
        collector.append(Y_test[i])
    counter -= 1
    if counter == 0:
        counter = cd

collector = np.array(collector)

# --

plt.plot(np.arange(len(Y_test)), Y_test, c='gray')
plt.xticks(np.arange(0, len(Y_test), cd))
plt.yticks(np.arange(0, max(Y_test), 10))
plt.grid(True)
plt.rc('axes', labelsize=4)
plt.savefig(root_path + 'result/real_trend.png')
plt.show()

# -

plt.plot(np.arange(len(collector)), collector, c='gray')
plt.xticks(np.arange(0, len(collector), 1))
plt.yticks(np.arange(0, max(Y_test), 10))
plt.grid(True)
plt.rc('axes', labelsize=4)
plt.savefig(root_path + 'result/real_trend_%sh.png' % cd)
plt.show()
