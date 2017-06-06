# _*_ coding: utf-8 _*_

import os
import xlrd
import json
import numpy as np


def save(data, file_name, path, type):
    if path[-1] != '/':
        destination = path + "/" + file_name
    else:
        destination = path + file_name
    f = open(destination, type)
    f.write(data)
    f.close()
    print("Saved.")


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
            elem = elem.encode(encoding='utf-8')
            if len(elem) > 0:
                data.append(elem.replace('"', '').replace('\r\n', '').split(','))

        f.close()
        return data
    else:
        print("This file doesn't exist.")


def load_xls(file_name, SheetName, path):
    if file_name != '':
        source = path + "/" + file_name
    else:
        source = path

    if os.path.exists(source):
        data = []
        workbook = xlrd.open_workbook(source)
        worksheet = workbook.sheet_by_name(SheetName)

        for rownum in xrange(worksheet.nrows):
            data.append(list(x.encode('utf-8') if type(x) == type(u'') else x for x in worksheet.row_values(rownum)))

        return data
    else:
        print("This xls file doesn't exist.")
# def load_xls(file_name, path):
#     if file_name != '':
#         source = path + "/" + file_name
#     else:
#         source = path
#
#     if os.path.exists(source):
#         sheets = json.dumps(get_data(source))
#         data = []
#         for line in (sheets.replace('{', '').replace('}', '').replace('Sheet1', '').replace(':', '').replace('"', '')
#                           .replace('[', '').replace(']]', ']').split('],')):
#             for element in line:
#                 # if element.find('\\u') != -1:
#                 #     element = element
#                 # print(element, end='')
#                 data.append(element.split(','))
#
#         return data
#     else:
#         print("This xls file doesn't exist.")


def load_all(data, path):
    if os.path.exists(path):
        if os.path.isdir(path):
            filelist = os.listdir(path)

            for f in filelist:
                    load_all(data, os.path.join(path, f))

        elif os.path.isfile(path):
            buffer = []
            if path.find('.xls') != -1:
                index = path.rfind('/')
                print(path[index+1:])
                buffer = load_xls('', 'Sheet1', path)

            # if path.find('HOURLY') != -1:
            #     index = path.rfind('/')
            #     print(path[index+1:])
            #     buffer = load('', path)

            if path.find('.csv') != -1:
                index = path.rfind('/')
                print(path[index + 1:])
                buffer = load('', path)

            if len(buffer) != 0:
                data.append(buffer)

        else:
            print("Loading error.")

    else:
        print("This path doesn't exist.")
