# _*_ coding: utf-8 _*_

from data_reader import data_reader
from config import root

root_path = root()


num_of_pollution_data = 21
num_of_weather_data = 13

# pollution_site_map = {
#     '中部': {'台中': ['大里', '忠明', '沙鹿', '西屯', '豐原'],
#            '南投': ['南投', '竹山'],
#            '彰化': ['二林', '彰化']},
#
#     '北部': {'台北': ['中山', '古亭', '士林', '松山', '萬華'],
#            '新北': ['土城', '新店', '新莊', '板橋', '林口', '汐止', '菜寮', '萬里'],
#            '基隆': ['基隆'],
#            '桃園': ['大園', '平鎮', '桃園', '龍潭']},
#
#     '宜蘭': {'宜蘭': ['冬山', '宜蘭']},
#
#     '竹苗': {'新竹': ['新竹', '湖口', '竹東'],
#            '苗栗': ['三義', '苗栗']},
#
#     '花東': {'花蓮': ['花蓮'],
#            '台東': ['臺東']},
#
#     '北部離島': {'彭佳嶼': []},
#
#     '西部離島': {'金門': ['金門'],
#              '連江': ['馬祖'],
#              '東吉嶼': [],
#              '澎湖': ['馬公']},
#
#     '雲嘉南': {'雲林': ['崙背', '斗六'],
#             '台南': ['善化', '安南', '新營', '臺南'],
#             '嘉義': ['嘉義', '新港', '朴子']},
#
#     '高屏': {'高雄': ['仁武', '前金', '大寮', '小港', '左營', '林園', '楠梓', '美濃'],
#            '屏東': ['屏東', '恆春', '潮州']}
# }

weather_site_map = {
    '中部': {'台中': ['台中', '梧棲'],
           '南投': ['日月潭'],
           '彰化': []},

    '北部': {'台北': ['台北', '竹子湖', '鞍部', '大屯山'],
           '新北': ['板橋', '淡水'],
           '基隆': [],
           '桃園': []},

    '宜蘭': {'宜蘭': ['宜蘭', '蘇澳']},

    '竹苗': {'新竹': ['竹北'],
           '苗栗': []},

    '花東': {'花蓮': ['花蓮'],
           '台東': ['台東', '成功', '大武', '蘭嶼']},

    '北部離島': {'彭佳嶼': ['彭佳嶼']},

    '西部離島': {'金門': [],
             '連江': [],
             '東吉嶼': ['東吉島'],
             '澎湖': ['澎湖']},

    '雲嘉南': {'雲林': [],
            '台南': ['台南', '永康'],
            '嘉義': ['阿里山', '玉山', '嘉義']},

    '高屏': {'高雄': ['高雄'],
           '屏東': ['恆春']}
}

weather_site_name2no = {
    '台中': 20, '梧棲': 30, '大坑': 158, '梨山': 170, '思源': 173,
    '日月潭': 28, '合歡山莊': 39, '神木村': 143, '鳳凰': 152, '竹山': 154, '廬山': 163, '昆陽': 165, '合歡山': 241, '中興新村': 242,
    '彰師大': 36, '員林': 160, '鹿港': 164,

    '大屯山': 1, '鞍部': 5, '台北': 6, '竹子湖': 7, '信義': 221, '南港': 224, '大直': 227, '內湖': 228, '士林': 229, '大崙尾山': 230, '社子': 231, '石碑': 232, '天母': 233, '五指山': 234,
    '五分山': 2, '板橋': 3, '淡水': 4, '龍洞': 33, '新店': 37, '福山': 195, '桶後': 201, '大豹': 205, '四堵': 206, '屈尺': 208, '坪林': 209, '泰平': 212, '山佳': 213, '三貂角': 215, '永和': 216, '福隆': 219, '雙溪': 222, '大尖山': 225, '三重': 226, '鼻頭角': 235, '金山': 238, '三和': 239, '富貴角': 240,
    '基隆': 8,
    '拉拉山': 34, '武陵': 38, '新屋': 210, '大坪': 236,

    '蘇澳': 11, '宜蘭': 12, '南澳': 176, '太平山': 179, '東澳': 180, '礁溪': 198, '龜山島': 200,

    '竹北': 24, '竹東': 191,
    '三義': 174, '苑里': 175, '觀霧': 178, '南庄': 182, '梅花': 185, '玉蘭': 186, '羅東': 187, '竹南': 189,

    '花蓮': 10, '太魯閣': 35, '玉里': 137, '佳心': 139, '舞鶴': 141, '豐濱': 145, '光復': 149, '加路蘭山': 150, '鳳林山': 153, '水璉': 155, '月眉山': 156, '鯉魚潭': 159, '水源': 161, '新城': 162, '富世': 167, '大禹嶺': 168, '天祥': 169, '和中': 171, '靜浦': 244,
    '大武': 22, '成功': 26, '蘭嶼': 27, '台東': 29, '南田': 109, '大溪山': 112, '金崙': 114, '太麻里': 115, '綠島': 116, '知本': 117, '紅葉山': 122, '鹿野': 123, '東河': 124, '紅石': 125, '池上': 128, '向陽': 132, '長濱': 134,

    '彭佳嶼': 9,

    '金門': 13,
    '馬祖': 32, '東沙': 43,
    '東吉島': 14,
    '澎湖': 15, '吉貝': 41,

    '宜梧': 144, '草嶺': 146, '四湖': 147, '虎尾': 151, '台西': 157,
    '台南': 16, '永康': 17, '七股': 40, '善化': 127, '玉井': 129, '佳里': 130, '曾文': 131, '新營': 135, '關子嶺': 138, '南化': 243,
    '嘉義': 19, '阿里山': 21, '玉山': 23, '馬頭山': 136, '奮起湖': 142, '大埔': 148,

    '高雄': 18, '古亭坑': 120, '美濃': 121, '甲仙': 126, '表湖': 133,
    '恆春': 25, '墾丁': 42, '貓鼻頭': 101, '鵝鑾鼻': 102, '佳樂水': 104, '檳榔': 105, '牡丹池山': 106, '楓港': 107, '牡丹': 108, '琉球嶼': 110, '枋寮': 111, '潮州': 113, '三地門': 118, '尾寮山': 119
}

weather_site_no2name = {y: x for x, y in weather_site_name2no.iteritems()}


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
        input("THis pollution(%s) hasn't been recorded." % pollution)
        # None


def read_data_sets(sites=['中山', '古亭', '士林', '松山', '萬華'], date_range=['2014', '2015'],
                   feature_selection=['PM2.5'], beginning='1/1', finish='12/31',
                   path=root_path+'dataset/', update=False):
    # print('Reading data .. ')

    y_d_h_data = data_reader(int(date_range[0]), int(date_range[-1]), path, update)

    # print('Reading data .. ok')
    # print('Date Range: ', date_range)
    # print('Construct feature vectors: ')
    num_of_missing = 0.
    total_number = 0.
    feature_vector_set = []
    for each_year in date_range:
        print('%s .. ok' % each_year)
        # for each_date in y_d_h_data[each_year]:
        days = 0
        for month in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']:
            # -- duration --
            if each_year == date_range[0] and int(month) < int(beginning[:beginning.index('/')]):  # start
                continue
            elif each_year == date_range[-1] and int(month) > int(finish[:finish.index('/')]):  # dead line
                continue
            # --
            if (month == '1') or (month == '3') or (month == '5') or (month == '7') or (
                        month == '8') or (month == '10') or (month == '12'):
                days = 31
            elif (month == '4') or (month == '6') or (month == '9') or (month == '11'):
                days = 30
            elif month == '2':
                if '2/29' in y_d_h_data[each_year]:
                    days = 29
                else:
                    days = 28

            for day in range(days):
                each_date = month + '/' + str(day + 1)

                # -- duration --
                if (each_year == date_range[0]) and (int(month) == int(beginning[:beginning.index('/')])) and (
                            (day+1) < int(beginning[(beginning.index('/')+1):])):  # start
                    continue
                elif (each_year == date_range[-1]) and int(month) == int(finish[:finish.index('/')]) and (
                            (day+1) > int(finish[(finish.index('/')+1):])):  # dead line
                    continue
                # --

                if not ('pollution' in y_d_h_data[each_year][each_date]):
                    print('Data of pollution missing: %s/%s' % (each_year, each_date))
                else:
                    for each_hour in range(24):
                        feature_vector = []
                        for site in sites:
                            if not (site in y_d_h_data[each_year][each_date]['pollution']):
                                # print('Data of site(%s) missing: %s/%s %d:00' % (site, each_year, each_date, each_hour))
                                for feature_elem in feature_selection:
                                    feature_vector.append('NaN')
                                    num_of_missing += 1
                                    total_number += 1
                            else:
                                for feature_elem in feature_selection:
                                    try:
                                        feature_vector.append(float(y_d_h_data[each_year][each_date]['pollution'][site][each_hour][pollution_to_pollution_no(feature_elem)]))
                                        total_number += 1
                                    except:
                                        # print('Data of feature(%s) of site(%s) missing: %s/%s %d:00' % (
                                        #     feature_elem, site, each_year, each_date, each_hour))
                                        feature_vector.append('NaN')
                                        num_of_missing += 1
                                        total_number += 1
                        feature_vector_set.append(feature_vector)

    # print('data_frame .. ok')
    print('Missing rate: %.5f' % (num_of_missing/total_number))
    return feature_vector_set


def concatenate_time_steps(X, n_steps):
    # input X should be a 2d-array
    # output Y will be a 3d-array
    # remainder element will be ignored
    # y are the elements of Y
    length = len(X)
    Y = []
    for i in range(length):
        y = []
        if (i + n_steps) <= length:
            for j in range(n_steps):
                y += list(X[i+j])
            Y.append(y)
    return Y


def construct_time_steps(X, n_steps):
    # input X should be a 2d-array
    # output Y will be a 3d-array
    # y are the elements of Y
    length = len(X)
    Y = []
    for i in range(length):
        y = []
        if (i + n_steps) <= length:
            for j in range(n_steps):
                y.append(X[i+j])
            Y.append(y)
    return Y
