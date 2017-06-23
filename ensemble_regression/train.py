# _*_ coding: utf-8 _*_

import os

fold = '/media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/python/weather_prediction/ensemble_regression/'
filename = 'cnn_model.py'
training_year = '2014-2016'
testing_year = '2016-2016'
training_duration = '1/1-10/31'
testing_duration = '11/1-12/31'
is_training = True
local = '北部'
city = '新北'
strategylist = ['highest', 'shift']

for strategy in strategylist:
    print('strategy: %s' % strategy)
    # for site in ['板橋', '林口']:
    #     for interval in [24]:
    #         # python $file_path $local $city $site $training_year $testing_year $training_duration $testing_duration $interval $is_training
    #         os.system("python %s %s %s %s %s %s %s %s %s %s %s"
    #                   % (fold+filename, local, city, site, training_year, testing_year, training_duration,
    #                      testing_duration, interval, is_training, strategy))
    city = '台北'
    for site in ['萬華']:  # ['中山', '古亭', '士林', '松山', '萬華']:
        for interval in [24]:
            os.system("python %s %s %s %s %s %s %s %s %s %s %s"
                      % (fold+filename, local, city, site, training_year, testing_year, training_duration,
                         testing_duration, interval, is_training, strategy))
