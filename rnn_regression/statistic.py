# _*_ coding: utf-8 _*_
import os

from file_reader import load


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


def load(file_name, path):
    if file_name != '':
        source = path + "/" + file_name
    else:
        source = path

    if os.path.exists(source):
        f = open(source, 'rb')
        data = []
        for i in f:
            elem = []
            try:
                elem = i.decode('utf-8')
            except:
                try:
                    elem = i.decode('big5')
                except:
                    try:
                        elem = i.decode('x-windows-950')
                    except:
                        elem = i.decode('ISO-8859-1')
            # elem = elem.encode(encoding='utf-8')
            if len(elem) > 0:
                data.append(elem.replace('\n', '').replace(':', ',').split(','))

        f.close()
        return data
    else:
        print("This file doesn't exist.")


def load_all(path, sites):
    if os.path.exists(path):
        if os.path.isdir(path):
            filelist = os.listdir(path)
            filelist.sort()

            for f in filelist:
                    load_all(os.path.join(path, f), sites)

        elif os.path.isfile(path):
            buffer = []

            if path.find('.ods') != -1:
                index = path.rfind('/')
                fileName = path[index + 1:]
                siteName = fileName[:fileName.index('_')]
                duration = fileName[fileName.find('_')+1:fileName.rfind('_')]
                interval = fileName[fileName.rfind('e')+1:fileName.index('.')]
                month = fileName[fileName.rfind('m')+1:fileName.rfind('_')]
                print('site name: %s, duration: %s, interval: %s' % (siteName, duration, interval))
                buffer = load('', path)

                if not(siteName in sites):
                    sites[siteName] = dict()
                if not(month in sites[siteName]):
                    sites[siteName][month] = dict()
                sites[siteName][month][interval] = dict()
                sites[siteName][month][interval]['accuracy'] = buffer[2][1]
                sites[siteName][month][interval]['rmse'] = float(buffer[0][1])
                sites[siteName][month][interval]['recall'] = float(buffer[6][2]) / (
                    float(buffer[6][4]) + float(buffer[6][2])) if int(float(buffer[6][4]) + float(buffer[6][2])) != 0 else -1
                sites[siteName][month][interval]['precision'] = float(buffer[6][2]) / (
                    float(buffer[6][3]) + float(buffer[6][2])) if int(float(buffer[6][3]) + float(buffer[6][2])) != 0 else -1
                sites[siteName][month][interval]['f1'] = (2 * sites[siteName][month][interval]['precision'] *
                                                          sites[siteName][month][interval]['recall']) / (
                                                         sites[siteName][month][interval]['precision'] +
                                                         sites[siteName][month][interval]['recall']) if (
                                                         sites[siteName][month][interval]['precision'] +
                                                         sites[siteName][month][interval]['recall']) != 0 else -1
                # sites[siteName][month][interval]['recall'] = float(buffer[8][2]) / (
                #     float(buffer[8][4]) + float(buffer[8][2])) if int(float(buffer[8][4]) + float(buffer[8][2])) != 0 else -1
                # sites[siteName][month][interval]['precision'] = float(buffer[8][2]) / (
                #     float(buffer[8][3]) + float(buffer[8][2])) if int(float(buffer[8][3]) + float(buffer[8][2])) != 0 else -1
                # sites[siteName][month][interval]['f1'] = (2 * sites[siteName][month][interval]['precision'] *
                #                                           sites[siteName][month][interval]['recall']) / (
                #                                         sites[siteName][month][interval]['precision'] +
                #                                         sites[siteName][month][interval]['recall']) if (
                #                                         sites[siteName][month][interval]['precision'] +
                #                                         sites[siteName][month][interval]['recall']) != 0 else -1

        else:
            print("Loading error.")

    else:
        print("This path doesn't exist.")


def ave(data, sites, months, hr, pred):
    value_ave = 0.
    total_num = 0
    for site in sites:
        if site in data:
            for month in months:
                if month in data[site]:
                    value_ave += data[site][month][hr][pred]
                    total_num += 1
    if total_num:
        return value_ave/total_num
    else:
        return -1


local = ''
city = ''
site_list = ['小港']  # 板橋 林口 萬華 新竹 苗栗 忠明 西屯 斗六 嘉義 左營 小港    # ['中山', '古亭', '士林', '松山', '萬華'] not used
month_list = ['1']  # ['1', '12']
hr = '12'
# pred = 'precision'  # precision, recall, f1
found_flag = 0

for local_element in pollution_site_map.keys():
    for city_element in pollution_site_map[local_element].keys():
        if site_list[0] in pollution_site_map[local_element][city_element]:
            local = local_element
            city = city_element
            found_flag = 1
            break
    if found_flag:
        break


path = ('/home/clliao/workspace/python/weather_prediction/rnn_regression/result/%s/%s/2017_m1_top1' % (local, city))
sites = dict()
load_all(path, sites)
print('Finish')

print('sites: %s, months: %s, hr: %s' % (site_list[0], month_list, hr))
print('rmse: %.4f' % ave(sites, site_list, month_list, hr, 'rmse'))
print('precision: %.4f' % ave(sites, site_list, month_list, hr, 'precision'))
print('recall: %.4f' % ave(sites, site_list, month_list, hr, 'recall'))
print('f1 score: %.4f' % ave(sites, site_list, month_list, hr, 'f1'))
