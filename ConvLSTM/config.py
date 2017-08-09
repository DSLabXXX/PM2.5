# _*_ coding: utf-8 _*_
import os

#root_path = '/media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/python/weather_prediction/ensemble_regression/'
root_path = os.getcwd() + '/ConvLSTM/'
data_path = os.getenv("HOME") + '/Dataset/AirQuality/'


def root():
    return root_path


def dataset_path():
    return data_path


def check_folder(file_path):
    # check folder
    if not os.path.isdir(data_path + 'cPickle/'):
        os.makedirs(data_path + 'cPickle/')

class Cite:
    def __init__(self, local_name, city_name, site_name, shape, adjacent_map):
        self.local = local_name     # '北部'
        self.city = city_name       # '台北'
        self.site_name = site_name  # '古亭'
        self.shape = shape  # two-tuple, i.e., (5, 5)
        self.adj_map = adjacent_map  # dict, i.e., {(3, 3):'古亭', (3, 2):'萬華', ...}

pollution_site_map2 = {
    # -------------------------------------------
    # |       |  菜寮  |  中山  |       |        |
    # -------------------------------------------
    # |  新莊  |       |  萬華  |  松山  |  汐止  |
    # -------------------------------------------
    # |       |  板橋  |  古亭  |       |        |
    # -------------------------------------------
    # |       |  土城  |  新店  |       |        |
    # -------------------------------------------
    # |       |        |       |       |        |
    # -------------------------------------------
    '古亭': Cite('北部', '台北', '古亭', (5, 5),
               {'菜寮': (0, 1), '中山': (0, 2),
                '新莊': (1, 0), '萬華': (1, 2), '松山': (1, 3), '汐止': (1, 4),
                '板橋': (2, 1), '古亭': (2, 2),
                '土城': (3, 1), '新店': (3, 2)})
}


pollution_site_map = {
    '中部': {'台中': ['大里', '忠明', '沙鹿', '西屯', '豐原'],  # 5
           '南投': ['南投', '竹山'],  # 2
           '彰化': ['彰化', '二林']},  # 2

    '北部': {'台北': ['中山', '古亭', '士林', '松山', '萬華'],  # 5
           '新北': ['林口', '土城', '新店', '新莊', '板橋', '汐止', '菜寮', '萬里'],  # 8
           '基隆': ['基隆'],  # 1
           '桃園': ['桃園', '大園', '平鎮', '龍潭']}, # 4

    '宜蘭': {'宜蘭': ['宜蘭', '冬山']},  # 2

    '竹苗': {'新竹': ['新竹', '湖口', '竹東'],  # 3
           '苗栗': ['苗栗', '三義']},  # 2

    '花東': {'花蓮': ['花蓮'],  # 1
           '台東': ['臺東']},  # 1

    '北部離島': {'彭佳嶼': []},

    '西部離島': {'金門': ['金門'], # 1
             '連江': ['馬祖'],  # 1
             '東吉嶼': [],
             '澎湖': ['馬公']},  # 1

    '雲嘉南': {'雲林': ['崙背', '斗六', '竹山'],  # 3
            '台南': ['臺南', '善化', '安南', '新營'],  # 4
            '嘉義': ['嘉義', '新港', '朴子']},  # 3

    '高屏': {'高雄': ['左營', '仁武', '前金', '大寮', '小港', '林園', '楠梓', '美濃'],  # 8
           '屏東': ['屏東', '恆春', '潮州']}  # 3
}


def site_map():
    return pollution_site_map


def site_map2():
    return pollution_site_map2
