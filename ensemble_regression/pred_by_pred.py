# _*_ coding: utf-8 _*_

from __future__ import print_function

import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

from file_reader import load
from data_reader import data_reader
from feature_processor import data_coordinate_angle
from missing_value_processer import missing_check
from config import root

root_path = root()

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


def observate(db, kind='rh', aims=['000', '024']):
    real = list()
    predict = list()
    for date in sorted(db.keys()):
        real.append(db[date][aims[0]][kind])
        predict.append(db[date][aims[-1]][kind])
    return real, predict

local = '中部'
city = '台中'
site_list = pollution_site_map[local][city]  # ['中山', '古亭', '士林', '松山', '萬華']
target_site = '西屯'
training_year = '2016'

all_sites_id_list_raw_data = load('EPA_81_target_stationlist.txt', root_path + 'dataset/中央氣象局-氣象監測站')
all_sites_id_list = dict()

all_sites_id_list_raw_data.pop(0)
for i in all_sites_id_list_raw_data:
    line_clear = i[0]
    while '  ' in line_clear:
        line_clear = line_clear.replace('  ', ' ')
    line_elem = line_clear.split(' ')
    all_sites_id_list[line_elem[5]] = line_elem[1]
# print()

y_d_h_data = data_reader(int(training_year), int(training_year), path=root_path+'dataset/', update=False)

raw_data = dict()
for i in site_list:
    raw_data[i] = dict()
    for j in ['000', '003', '006', '009', '012', '015', '018', '021', '024']:
        file_name = '%s_%s_%s.txt' % (all_sites_id_list[i], training_year, j)
        print(file_name)
        site_data = load(file_name, root_path + 'dataset/中央氣象局-氣象監測站/wrf_d02_2016_epa/')
        site_data.pop(0)

        for k in site_data:
            line_clear = k[0]
            while '  ' in line_clear:
                line_clear = line_clear.replace('  ', ' ')
            line_elem = line_clear.split(' ')

            if not (line_elem[0] in raw_data[i]):
                raw_data[i][line_elem[0]] = dict()

            raw_data[i][line_elem[0]][j] = dict()
            raw_data[i][line_elem[0]][j]['t'] = float(line_elem[1])
            raw_data[i][line_elem[0]][j]['rh'] = float(line_elem[2])
            raw_data[i][line_elem[0]][j]['ws'] = float(line_elem[5])
            raw_data[i][line_elem[0]][j]['wd'] = float(line_elem[6])

            if j == '000':
                year = line_elem[0][0:4]
                month = int(line_elem[0][4:6])
                day = int(line_elem[0][6:])
                try:
                    raw_data[i][line_elem[0]][j]['PM2.5'] = y_d_h_data[year][str(month)+'/'+str(day)]['pollution'][i][8][4]
                except:
                    raw_data[i][line_elem[0]][j]['PM2.5'] = 'NaN'
                try:
                    raw_data[i][line_elem[0]][j]['O3'] = y_d_h_data[year][str(month)+'/'+str(day)]['pollution'][i][8][2]
                except:
                    raw_data[i][line_elem[0]][j]['O3'] = 'NaN'

                # print()

x_train = list()
y_train = list()
# pollution_kind = ['PM2.5', 'O3', 't', 'rh', 'ws', 'wd']


def missing_mark(unmark, vector, marked='NaN'):
    while unmark in vector:
        vector[vector.index(unmark)] = marked
    return vector

for i in sorted(raw_data[site_list[0]].keys()):
    x_train_line = list()
    for j in site_list:
        for k in ['000', '003', '006', '009', '012', '015', '018', '021', '024']:
            if k == '000':
                if j == target_site:
                    try:
                        if raw_data[j][i][k]['PM2.5'] == 'NaN':
                            y_train.append(raw_data[j][i][k]['PM2.5'])
                        else:
                            y_train.append(float(raw_data[j][i][k]['PM2.5']))
                    except:
                        y_train.append('NaN')
                try:
                    x_train_line.append(float(raw_data[j][i][k]['PM2.5']))
                except:
                    x_train_line.append('NaN')
                try:
                    x_train_line.append(float(raw_data[j][i][k]['O3']))
                except:
                    x_train_line.append('NaN')
            x_train_line.append(raw_data[j][i][k]['t'])
            x_train_line.append(raw_data[j][i][k]['rh'])
            x_train_line.append(raw_data[j][i][k]['ws'])
            wd = data_coordinate_angle(raw_data[j][i][k]['wd'])
            x_train_line.append(wd[0])
            x_train_line.append(wd[1])
    x_train.append(missing_mark(-9999, x_train_line))

# print('Imputation .. ')
x_train = missing_check(x_train)
y_train = missing_check(missing_mark(-9999, y_train))
y_train = np.reshape(y_train, [len(y_train)])

# print()

x_train = x_train[:-1]
y_train = y_train[1:]

# print()

test_ratio = len(x_train)/10

x_train = np.array(x_train[:-test_ratio])
y_train = np.array(y_train[:-test_ratio])

x_test = np.array(x_train[-test_ratio:])
y_test = np.array(y_train[-test_ratio:])

print('Normalize ..')

mean_x_train = np.mean(x_train, axis=0)
std_x_train = np.std(x_train, axis=0)
if 0 in std_x_train:
    input("Denominator can't be 0.")
x_train = np.array([(x-mean_x_train)/std_x_train for x in x_train])
X_test = np.array([(x-mean_x_train)/std_x_train for x in x_test])

mean_y_train = np.mean(y_train)
std_y_train = np.std(y_train)
if not std_y_train:
    input("Denominator can't be 0.")
y_train = [(y - mean_y_train) / std_y_train for y in y_train]
print('mean_y_train: %f  std_y_train: %f' % (mean_y_train, std_y_train))

print('Training .. ')

xgb_model = xgb.XGBRegressor().fit(x_train, y_train)
xgb_pred = xgb_model.predict(x_train)

pred = mean_y_train + std_y_train * xgb_pred
y_train = mean_y_train + std_y_train * np.array(y_train)
lose = np.mean((y_train - pred)**2, 0)**0.5
print('lose: %.5f' % (lose))

# plt.plot(np.arange(len(pred)), y_train[:len(pred)], c='gray')
# plt.plot(np.arange(len(pred)), pred, color='pink')
# plt.xticks(np.arange(0, len(pred), 24))
# plt.yticks(np.arange(0, max(y_train), 10))
# plt.grid(True)
# plt.rc('axes', labelsize=4)
# plt.show()

print('Testing ..')

xgb_pred = xgb_model.predict(x_test)
pred = mean_y_train + std_y_train * xgb_pred
print('rmse: %.5f' % (np.mean((y_test - pred)**2, 0)**0.5))

plt.plot(np.arange(len(pred)), y_test[:len(pred)], c='gray')
plt.plot(np.arange(len(pred)), pred, color='pink')
plt.xticks(np.arange(0, len(pred), 24))
plt.yticks(np.arange(0, max(y_test), 10))
plt.grid(True)
plt.rc('axes', labelsize=4)
# plt.show()

print('Writing result ..')
with open(root_path + 'pred_by_pred_result/%s_%s.ods' % (target_site, training_year), 'wt') as f:
    print('lose: %.5f' % (lose), file=f)
    f.write('\n')
    print('rmse: %f' % (np.mean((y_test - pred)**2, 0)**0.5), file=f)
    f.write('\n')
    print('Writing result .. ok')
    plt.savefig(root_path + 'pred_by_pred_result/%s_%s.png' % (target_site, training_year), dpi=100)

# -------------
plt.close()
# -------------

for ob_target in ['t', 'rh', 'ws']:
    [real, predict] = observate(raw_data[target_site], kind=ob_target, aims=['000', '024'])

    real = real[1:]
    predict = predict[:-1]

    # -----

    real_clean = real
    predict_clean = predict
    while (-9999 in real_clean):
        del_index = real_clean.index(-9999)
        real_clean.pop(del_index)
        predict_clean.pop(del_index)

    while (-9999 in predict_clean):
        del_index = predict_clean.index(-9999)
        real_clean.pop(del_index)
        predict_clean.pop(del_index)

    # -----

    real = missing_mark(-9999, real, marked=-1)
    predict = missing_mark(-9999, predict, marked=-1)

    mean_real = np.mean(real_clean)
    std_real = np.std(real_clean)

    plt.plot(np.arange(len(real)), real, c='gray')
    plt.plot(np.arange(len(real)), predict, color='pink')
    plt.grid(True)
    plt.rc('axes', labelsize=4)

    real_clean = np.array(real_clean)
    predict_clean = np.array(predict_clean)

    with open(root_path + 'pred_by_pred_result/%s_%s_%s.ods' % (target_site, training_year, ob_target), 'wt') as f:
        print('rmse: %f' % (np.mean((real_clean - predict_clean)**2, 0)**0.5), file=f)
        f.write('\n')
        print('mean: %f' % (mean_real), file=f)
        f.write('\n')
        print('std: %f' % (std_real), file=f)
        f.write('\n')

        plt.savefig(root_path + 'pred_by_pred_result/%s_%s_%s.png' % (target_site, training_year, ob_target), dpi=100)

    plt.close()

print('Writing result .. ok')
