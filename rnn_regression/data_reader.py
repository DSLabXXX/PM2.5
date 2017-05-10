# _*_ coding: utf-8 _*_

from file_reader import load_all
from operator import itemgetter
from feature_processor import time_to_angle, data_coordinate_angle
import numpy as np
import cPickle
import os


root_path = '/home/clliao/workspace/python/weather_prediction/rnn_regression/'


def site_to_site_no(site_pient):
    site_pient = int(site_pient)
    if site_pient <= 43:
        return site_pient-1

    elif site_pient >= 101:
        return site_pient-101+44-1

    else:
        print("This site doesn't exist.")


def param_code_to_param_code_no(param_code):
    param_code = int(param_code)
    if param_code == 10:
        return 0
    elif param_code == 11:
        return 1
    elif param_code == 14:
        return 2
    elif param_code == 15:
        return 3
    elif param_code == 17:
        return 4
    elif param_code == 23:
        return 5
    elif param_code == 38:
        return 6
    elif param_code == 103:
        return 7
    elif param_code == 104:
        return 8
    elif param_code == 105:
        return 9
    elif param_code == 106:
        return 10
    elif param_code == 107:
        return 11
    elif param_code == 108:
        return 12
    else:
        print("This param_code doesn't exist.")


def pollution_to_pollution_no(pollution):
    if pollution == 'SO2':return 0
    elif pollution == 'CO':return 1
    elif pollution == 'O3':return 2
    elif pollution == 'PM10':return 3
    elif pollution == 'PM2.5':return 4
    elif pollution == 'NOx':return 5
    elif pollution == 'NO':return 6
    elif pollution == 'NO2':return 7
    elif pollution == 'THC':return 8
    elif pollution == 'NMHC':return 9
    elif pollution == 'CH4':return 10
    elif pollution == 'UVB':return 11
    elif pollution == 'AMB_TEMP':return 12
    elif pollution == 'RAINFALL':return 13
    elif pollution == 'RH':return 14
    elif pollution == 'WIND_SPEED':return 15
    elif pollution == 'WIND_DIREC':return 16
    elif pollution == 'WS_HR':return 17
    elif pollution == 'WD_HR':return 18
    elif pollution == 'PH_RAIN':return 19
    elif pollution == 'RAIN_COND':return 20
    else:
        # print("THis pollution(%s) hasn't been recorded." % pollution)
        None


# ----------  WORK START  ----------
def data_reader(start_year, last_year, path=root_path+'dataset/', update=False):
    y_d_h_data = dict()
    not_exit_flag = 0
    while (start_year != last_year+1) and (not update):
        if os.path.exists(path+'cPickle/pollution_and_weather_data_'+str(start_year)):
            not_exit_flag += 1
            print('Reading %d data by cPickle .. ' % start_year)
            fr = open(path+'cPickle/pollution_and_weather_data_'+str(start_year), 'rb')
            y_d_h_data[str(start_year)] = cPickle.load(fr)
            fr.close()
        start_year += 1

    if not_exit_flag > 0:
        return y_d_h_data

    elif not_exit_flag == 0:
        print('Start from reading raw data.')
        # feature_vector = []
        y_d_h_data = dict()  # years, days and hours, then pollution and weather data

        # ---------- pollution ----------
        pollution_data_files = []  # multi-files
        num_pollution_property = 21

        # --- csv ---
        # csv_pollution_data = []
        load_all(pollution_data_files, path+'Data_of_Air_Pollution/')

        # data pre-processing : format
        keep_date = ''
        pollution_vector_one_day = []

        for single_file_pollution_data in pollution_data_files:
            for line in single_file_pollution_data:
                if line == single_file_pollution_data[0]:
                    None
                else:
                    if line[0].find('-') != -1:
                        line[0] = line[0].replace('-0', '/')  # 2008-01-01 -> 2008/1/1
                        line[0] = line[0].replace('-', '/')  # 2008-10-12 -> 2008/10/12
                    if line[0].find('/0') != -1:
                        line[0] = line[0].replace('/0', '/')  # 2010/01/01 ->2010/1/1

                    year = line[0][:line[0].find('/')]
                    date = line[0][line[0].find('/')+1:]
                    # check/create year dict., ex: 2016, 2015
                    if not(year in y_d_h_data):
                        y_d_h_data[year] = dict()
                    # check/create date dict., ex: 01/01, 10/31
                    if not(date in y_d_h_data[year]):
                        y_d_h_data[year][date] = dict()
                    # pollution sites dict.
                    if not('pollution' in y_d_h_data[year][date]):
                        y_d_h_data[year][date]['pollution'] = dict()

                    print(line[:3])

                    if keep_date != line[0]:  # a new day
                        if keep_date != '' and (keep_date[:keep_date.find('/')] == year):
                            y_d_h_data[keep_date[:keep_date.find('/')]][keep_date[keep_date.find('/')+1:]]['pollution'][line[1]] = pollution_vector_one_day
                            pollution_vector_one_day = []
                        elif keep_date != '':
                            pollution_vector_one_day = []

                        keep_date = line[0]

                        # Reserve 'num_pollution_property' entries for data, and take '-' to mean missing value
                        for each_hour in np.arange(24):
                            pollution_vector_one_day.append(['-' for i in np.arange(num_pollution_property)])

                    for each_hour in np.arange(24):
                        # The first three elements are date, sites and kind of pollution
                        try:
                            pollution_vector_one_day[each_hour][pollution_to_pollution_no(line[2].replace(' ', ''))] = line[3+each_hour]
                        except:
                            break

                if line == single_file_pollution_data[-1]:  # the last recorded day of this file
                    y_d_h_data[year][date]['pollution'][line[1]] = pollution_vector_one_day

        print('--------------------------------------------------------------------------------------')

        # # ---------- weather ----------
        # num_weather_property = 13
        # # num_sites = 449
        #
        # weather_data = []
        # load_all(weather_data, path+'Data_of_Weather/')
        #
        # for file_i in np.arange(len(weather_data)):
        #     # sorting by date -> site -> param_code
        #     for line_j in np.arange(len(weather_data[file_i])):
        #         [year, _, date, angle] = time_to_angle(weather_data[file_i][line_j][2].replace(' 00:00:00', ''))
        #
        #         format_day_order = angle/360.
        #         weather_data[file_i][line_j].append(int(year) + format_day_order)
        #         weather_data[file_i][line_j][0] = int(weather_data[file_i][line_j][0])
        #
        #     weather_data[file_i] = sorted(weather_data[file_i], key=itemgetter(len(weather_data[file_i][line_j])-1, 0, 1))
        #
        #     print('Sorted complete.')
        #
        #     keep_date = ''
        #     keep_site = ''
        #     for line_j in np.arange(len(weather_data[file_i])):
        #         # a new site
        #         if weather_data[file_i][line_j][0] != keep_site:
        #             if keep_site != '':
        #                 y_d_h_data[year][date]['weather'][keep_site] = weather_vector
        #                 print(year + '/' + date + ': site- %s' % keep_site)
        #             keep_site = weather_data[file_i][line_j][0]
        #
        #             weather_vector = []
        #
        #
        #         # a new day
        #         if weather_data[file_i][line_j][2].replace(' 00:00:00', '') != keep_date:
        #             if keep_date != '' and len(weather_vector) != 0:
        #                 y_d_h_data[year][date]['weather'][keep_site] = weather_vector
        #                 print(year + '/' + date + ': site- %s' % keep_site)
        #
        #             keep_date = weather_data[file_i][line_j][2].replace(' 00:00:00', '')
        #             year = keep_date[:keep_date.find('/')]
        #             date = keep_date[keep_date.find('/')+1:]
        #
        #             # check/create year dict., ex: 2016, 2015
        #             if not (year in y_d_h_data):
        #                 y_d_h_data[year] = dict()
        #             # check/create date dict., ex: 1/1, 10/31
        #             if not (date in y_d_h_data[year]):
        #                 y_d_h_data[year][date] = dict()
        #             # weather sites dict.
        #             if not('weather' in y_d_h_data[year][date]):
        #                 y_d_h_data[year][date]['weather'] = dict()
        #
        #             weather_vector = []
        #
        #         # Initiate weather_vector, when 'a new day' or 'a new site'.
        #         if len(weather_vector) == 0:
        #             for each_hour in np.arange(24):
        #                 weather_vector.append(['-' for i in np.arange(num_weather_property)])
        #
        #         # collecting data
        #         for each_hour in np.arange(24):
        #             weather_vector[each_hour][param_code_to_param_code_no(weather_data[file_i][line_j][1])] \
        #                 = weather_data[file_i][line_j][3+each_hour]  # the first three element mean site, param_code and date
        #
        #         if line_j == len(weather_data[file_i])-1:  # the last day
        #             y_d_h_data[year][date]['weather'][keep_site] = weather_vector
        #     print('----')

        print('Saving .. ')
        for years in y_d_h_data.keys():
            fw1 = open(path+'cPickle/pollution_and_weather_data_'+years, 'wb')
            cPickle.dump(y_d_h_data[years], fw1)
            fw1.close()

        print('Saved.')

        return y_d_h_data
