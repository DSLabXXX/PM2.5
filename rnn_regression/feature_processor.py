import numpy as np


def time_to_angle(time):
    year = time[:time.find('/')]
    month = time[time.find('/')+1:time.rfind('/')]
    date = time[time.rfind('/')+1:]
    # print(year+'/'+month+'/'+date)

    # date to 365 day
    date = int(date)
    if month == '1':
        day = date
    elif month == '2':  # 1/31
        day = date + 31
        if date == 29:
            day -= 0.5
    elif month == '3':  # 2/28
        day = date + 31 + 28
    elif month == '4':  # 3/31
        day = date + 31 + 28 + 31
    elif month == '5':  # 4/30
        day = date + 31 + 28 + 31 + 30
    elif month == '6':  # 5/31
        day = date + 31 + 28 + 31 + 30 + 31
    elif month == '7':  # 6/30
        day = date + 31 + 28 + 31 + 30 + 31 + 30
    elif month == '8':  # 7/31
        day = date + 31 + 28 + 31 + 30 + 31 + 30 + 31
    elif month == '9':  # 8/31
        day = date + 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31
    elif month == '10':  # 9/30
        day = date + 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30
    elif month == '11':  # 10/31
        day = date + 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31
    elif month == '12':  # 11/30
        day = date + 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30

    try:
        angle = day / 365 * 360
    except:
        print(time)
        input('Please type in any key to continue.')

    return year, month, date, angle


def data_coordinate_angle(angle):
    while angle > 360:
        angle -= 360
    while angle < 0:
        angle += 360

    coordin = [np.cos(angle*np.pi/180.), np.sin(angle*np.pi/180.)]
    return coordin


def correlation(x, y):
    std_x = np.std(x)
    std_y = np.std(y)

    expect_x = np.mean(x)
    expect_y = np.mean(y)

    corr_xy = (np.mean((x - expect_x) * (y - expect_y))) / (std_x * std_y)
    return corr_xy
