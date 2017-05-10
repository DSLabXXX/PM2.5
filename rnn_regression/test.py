# _*_ coding: utf-8 _*_

# from __future__ import print_function
import json

with open('/home/clliao/workspace/python/weather_prediction/rnn_regression/dataset/Power.json') as json_data:
    d = json.load(json_data, encoding='utf-8')
    for i in d.keys():
        print(i)
        for j in d[i]:
            for k in j:
                print(k)  # print(k.decode('string_escape'))
print('--')
