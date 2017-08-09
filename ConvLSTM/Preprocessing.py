# reading data
print('Reading data .. ')
start_time = time.time()
initial_time = time.time()
print('preparing training set ..')

raw_data_train = read_data_sets(sites=site_list+[target_site], date_range=np.atleast_1d(training_year),
                                beginning=training_duration[0], finish=training_duration[-1],
                                feature_selection=pollution_kind, update=data_update)
raw_data_train = missing_check(raw_data_train)
Y_train = np.array(raw_data_train)[:, -len(pollution_kind):]
Y_train = Y_train[:, pollution_kind.index(target_kind)]
raw_data_train = np.array(raw_data_train)[:, :-len(pollution_kind)]

print('preparing testing set ..')

raw_data_test = read_data_sets(sites=site_list + [target_site], date_range=np.atleast_1d(testing_year),
                               beginning=testing_duration[0], finish=testing_duration[-1],
                               feature_selection=pollution_kind, update=data_update)
Y_test = np.array(raw_data_test)[:, -len(pollution_kind):]
Y_test = Y_test[:, pollution_kind.index(target_kind)]
raw_data_test = missing_check(np.array(raw_data_test)[:, :-len(pollution_kind)])
Y_test = np.array(Y_test, dtype=np.float)

final_time = time.time()
print('Reading data .. ok, ', end='')
time_spent_printer(start_time, final_time)