import numpy as np
#import matplotlib.pyplot as plt
from scipy.fftpack import fft


def time_spent_printer(start_time, final_time):
    spent_time = final_time - start_time
    print('totally spent ', end='')
    print(int(spent_time / 3600), 'hours ', end='')
    print(int((int(spent_time) % 3600) / 60), 'minutes ', end='')
    print((int(spent_time) % 3600) % 60, 'seconds')


# for interval
def avg(Y, interval_hours):
    reserve_hours = interval_hours - 1
    deadline = 0
    for i in range(len(Y)):
        # check the reserve data is enough or not
        if (len(Y) - i - 1) < reserve_hours:
            deadline = i
            break  # not enough
        for j in range(reserve_hours):
            Y[i] += Y[i + j + 1]
        Y[i] /= interval_hours
    if deadline:
        Y = Y[:deadline]
    return Y


#
# To find the mean of the K highest values within the next time interval
# Y: value of PM2.5
# K: the K highest values
#
def topK_next_interval(Y, interval_hours, K):
    if interval_hours > K:
        deadline = 0
        for i in range(len(Y)):
            # check the reserve data is enough or not
            if (len(Y) - i) < interval_hours:
                deadline = i
                break  # not enough
            higher_list = []
            for j in range(interval_hours):
                if len(higher_list) < K:
                    higher_list.append(Y[i + j])
                elif Y[i + j] > higher_list[0]:
                    higher_list[0] = Y[i + j]
                higher_list = sorted(higher_list)
            Y[i] = np.array(higher_list).sum() / K
        if deadline:
            Y = Y[:deadline]
    return np.array(Y)


# for interval
def higher(Y, interval_hours):
    reserve_hours = 1  # choose the first n number of biggest
    if interval_hours > reserve_hours:
        deadline = 0
        for i in range(len(Y)):
            # check the reserve data is enough or not
            if (len(Y) - i) < interval_hours:
                deadline = i
                break  # not enough
            higher_list = []
            for j in range(interval_hours):
                if len(higher_list) < reserve_hours:
                    higher_list.append(Y[i + j])
                elif Y[i + j] > higher_list[0]:
                    higher_list[0] = Y[i + j]
                higher_list = sorted(higher_list)
            Y[i] = np.array(higher_list).sum() / reserve_hours
        if deadline:
            Y = Y[:deadline]
    return Y


def smooth(Y, smooth_range=2):  # average with last and future hours
    New_Y = np.array(Y, dtype=float)
    for i in range(len(Y)):
        ave = 0.
        denominator = smooth_range * 2 + 1
        for j in range(smooth_range+1):
            if j == 0:
                ave += Y[i]
            else:
                if (i+j) < len(Y):
                    ave += Y[i + j]
                else:
                    denominator -= 1

                if (i-j) >= 0:
                    ave += Y[i - j]
                else:
                    denominator -= 1
        ave = ave/denominator
        New_Y[i] = ave
    return New_Y


def highest(Y, smooth_range=2):  # highest of last and future hours
    New_Y = np.array(Y, dtype=float)

    if Y.ndim == 3:
        for i in range(Y.shape[0]):
            for r in range(Y.shape[1]):
                for c in range(Y.shape[2]):
                    highest = 0.
                    for j in range(smooth_range + 1):
                        if j == 0:
                            highest = Y[i, r, c]
                        else:
                            if (i + j) < Y.shape[0]:
                                if highest < Y[i+j, r, c]:
                                    highest = Y[i+j, r, c]

                            if (i - j) >= 0:
                                if highest < Y[i-j, r, c]:
                                    highest = Y[i-j, r, c]
                    New_Y[i] = highest
    else:
        for i in range(len(Y)):
            highest = 0.
            # denominator = degree * 2 + 1
            for j in range(smooth_range+1):
                if j == 0:
                    highest = Y[i]
                else:
                    if (i+j) < len(Y):
                        if highest < Y[i + j]:
                            highest = Y[i + j]

                    if (i-j) >= 0:
                        if highest < Y[i - j]:
                            highest = Y[i - j]
            New_Y[i] = highest
    return New_Y


# def plotting(data, filename, root_path, grid=[24, 10], save=False, show=False, collor=['mediumaquamarine', 'pink', 'lavender']):
#     if len(grid) != 2:
#         print('len(grid) must equal to 2')
#     for i in range(len(data)):
#         c = i if i < len(collor) else i % len(collor)
#         plt.plot(np.arange(len(data[i])), data[i], c=collor[c])
#
#     plt.xticks(np.arange(0, len(data[0]), grid[0]))
#     plt.yticks(np.arange(0, max(data[0]), grid[1]))
#     plt.grid(True)
#     plt.rc('axes', labelsize=4)
#     if save:
#         plt.savefig(root_path + 'result/' + filename)
#     if show:
#         plt.show()


# -- highest & lowest & average
# Input:
#   tensor_4d: num_of_data * layer_1_time_steps * layer_2_time_steps * feature_vector
# Output:
#   new_features_tensor:
def h_l_avg(tensor_4d, pollution_kind, site_list, target_site, target_kind, feature_kind_shift):
    new_features_tensor = list()
    for f_i in range(len(tensor_4d)):
        new_features_matrix = list()

        length_of_kind_list = len(pollution_kind)
        index_of_site = site_list.index(target_site)
        length_of_kind_list = length_of_kind_list + 1 if 'WIND_DIREC' in pollution_kind else length_of_kind_list
        index_of_kind = pollution_kind.index(target_kind)
        index = feature_kind_shift + index_of_kind + index_of_site * length_of_kind_list

        for f_k in range(len(tensor_4d[f_i])):
            feature_vector = tensor_4d[f_i, f_k, :, index]

            new_features_vector = list()

            new_features_vector.append(np.max(feature_vector))
            new_features_vector.append(np.min(feature_vector))
            new_features_vector.append(np.average(feature_vector))

            new_features_matrix.append(new_features_vector)

        new_features_tensor.append(new_features_matrix)

    return new_features_tensor


#
# -- fourier transfer --
#
def time_domain_and_frequency_domain(input_tensor, pollution_kind, site_list, target_site, feature_kind_shift):
    freq_and_time_output = list()
    for f_i in range(len(input_tensor)):
        freq_and_time_tensor = list()

        length_of_kind_list = len(pollution_kind)
        index_of_site = site_list.index(target_site)
        length_of_kind_list = length_of_kind_list + 1 if 'WIND_DIREC' in pollution_kind else length_of_kind_list

        for f_j in pollution_kind:
            if f_j == 'WIND_DIREC':
                continue

            freq_feature_matrix = list()
            time_matrix = list()

            index_of_kind = pollution_kind.index(f_j)
            index = feature_kind_shift + index_of_kind + index_of_site * length_of_kind_list

            for f_k in range(len(input_tensor[f_i])):
                freq_feature_matrix.append(np.real(fft(input_tensor[f_i, f_k, :, index])))
                time_matrix.append(input_tensor[f_i, f_k, :, index])

            freq_and_time_tensor.append(np.array(freq_feature_matrix))
            freq_and_time_tensor.append(np.array(time_matrix))

        freq_and_time_output.append(freq_and_time_tensor)

    return freq_and_time_output


