# _*_ coding: utf-8 _*_

import random
import numpy as np
import tensorflow as tf

from generator import Generator
from dataloader import Gen_Data_loader, Dis_dataloader
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

local = '北部'
city = '台北'
site_list = pollution_site_map[local][city]  # ['中山', '古亭', '士林', '松山', '萬華']
target_site = '中山'

training_year = ['2014', '2016']  # change format from   2014-2015   to   ['2014', '2015']
testing_year = ['2017', '2017']

training_duration = ['1/1', '12/31']
testing_duration = ['1/1', '1/31']
# interval_hours = 6  # predict the label of average data of many hours later, default is 1

target_kind = 'PM2.5'
pollution_kind = ['PM2.5', 'O3', 'WIND_SPEED', 'WIND_DIREC']  # pollution_kind = ['PM2.5', 'O3', 'AMB_TEMP', 'RH', 'WIND_SPEED', 'WIND_DIREC']

is_training = False
data_update = False

#
#  Generator  Hyper-parameters
# ----------------------------------------------------------------------------
NUM_EMB = (len(site_list)*len(pollution_kind)+len(site_list)) if 'WIND_DIREC' in pollution_kind else (len(site_list)*len(pollution_kind))  # size of input vector
EMB_DIM = 32            # embedding dimension (second layer)
HIDDEN_DIM_1 = 32         # hidden state dimension of lstm cell of the first layer
HIDDEN_DIM_2 = 32         # hidden state dimension of lstm cell of the second layer
SEQ_LENGTH_1 = 12         # sequence length of the first layer
SEQ_LENGTH_2 = 24        # sequence length of the second layer
START_TOKEN = -1
# PRE_EPOCH_NUM = 1     # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64
# ROLLOUT_NUM = 16
# G_STEPS = 1
# ----------------------------------------------------------------------------

#
#  Discriminator  Hyper-parameters
# ----------------------------------------------------------------------------
# dis_embedding_dim = 64
# dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
# dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
# dis_dropout_keep_prob = 0.75
# dis_l2_reg_lambda = 0.2
# dis_batch_size = 64
# D_STEPS = 5
# ----------------------------------------------------------------------------

#
#  Basic Training Parameters
# ----------------------------------------------------------------------------
# TOTAL_BATCH = 800
positive_file = 'save/real_data.txt'
negative_file = 'save/generator_sample.txt'
# eval_file = 'save/eval_file.txt'
# generated_num = 10000
# ----------------------------------------------------------------------------


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    #
    # Declare data loader
    # ----------------------------------------------------------------------------
    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    # likelihood_data_loader = Gen_Data_loader(BATCH_SIZE)  # For testing
    dis_data_loader = Dis_dataloader(BATCH_SIZE)
    # ----------------------------------------------------------------------------

    # load the air data and write positive file
    # gen_data_loader.load_data(root_path+positive_file, site_list, target_site, target_kind, training_year,
    #                           training_duration, pollution_kind, SEQ_LENGTH)

    #
    # Declare Generator & Discriminator
    # ----------------------------------------------------------------------------
    # declare: generator
    generator = Generator(NUM_EMB, EMB_DIM, BATCH_SIZE, HIDDEN_DIM_1, HIDDEN_DIM_2, SEQ_LENGTH_1, SEQ_LENGTH_2, START_TOKEN)

    print('OK')


if __name__ == '__main__':
    main()
