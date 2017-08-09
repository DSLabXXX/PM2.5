import time

import keras
from keras.layers import Input, Conv2D, LSTM, MaxPooling2D, concatenate, Bidirectional, Activation, TimeDistributed, Dense, Dropout, Flatten, ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping

from reader import read_data_map, construct_time_map
from missing_value_processer import missing_check
from config import root, dataset_path, site_map2
from Utilities import *


#
# ----- START: Parameters Declaration ----------------------------------------------------------------------------------
#

# Define target duration for training
training_year = [2014, 2016]  # change format from   2014-2015   to   ['2014', '2015']
training_duration = ['1/1', '12/31']

# Define target duration for prediction
testing_year = [2017, 2017]
testing_duration = ['1/1', '1/31']
interval_hours = 12  # predict the average of the subsequent hours as the label, set to 1 as default.
is_training = True  # True False

# Define target site and its adjacent map for prediction
pollution_site_map2 = site_map2()
target_site = pollution_site_map2['古亭']
center_i = int(target_site.shape[0]/2)
center_j = int(target_site.shape[1]/2)
local = target_site.local
city = target_site.city
target_site_name = target_site.site_name
site_list = list(target_site.adj_map.keys())  # ['士林', '中山', '松山', '汐止', '板橋', '萬華', '古亭', '土城', '新店']

# Define data loading parameters
data_update = False
pollution_kind = ['PM2.5', 'O3', 'AMB_TEMP', 'RH', 'WIND_SPEED', 'WIND_DIREC']
target_kind = 'PM2.5'

# Define pre-processing parameters
feature_kind_shift = 6  # 'day of year', 'day of week' and 'time of day' respectively are represented by two dimensions
smooth_range = 6
plot_grid = [interval_hours, 10]

# Define model parameters
regularizer = 1e-6
batch_size = 128
num_filters = 32
kernel_size = (3, 3)
pool_size = (2, 2)
num_classes = 10
epoch = 50
rnn_dropout = 0.5
r_dropout = 0.5
cnn_dropout = 0.25
dnn_dropout = 0.5
train_seg_length = 24
output_size = 1

#
# ----- END: Parameters Declaration ------------------------------------------------------------------------------------
#


#
# ----- START: Year Processing -----------------------------------------------------------------------------------------
#

# Clear redundant year, i.e., [2014, 2014] ==> [2014]
if training_year[0] == training_year[1]:
    training_year.pop(1)
if testing_year[0] == testing_year[1]:
    testing_year.pop(1)
else:
    input('The range of testing year should not be more than one year or cross contiguous years.')

# Generate years sequence, i.e., [2014, 2016] ==> [2014, 2015, 2016]
range_of_year = training_year[-1] - training_year[0]
for i in range(range_of_year):
    if not(int(i + training_year[0]) in training_year):
        training_year.insert(i, int(i + training_year[0]))

#
# ----- END: Year Processing -------------------------------------------------------------------------------------------
#


#
# ----- START: Data Loading --------------------------------------------------------------------------------------------
#

# Set the path of training & testing data
root_path = root()
data_path = dataset_path()
testing_month = testing_duration[0][:testing_duration[0].index('/')]
folder = root_path+"model/%s/%s/%sh/" % (local, city, interval_hours)
training_start_point = training_duration[0][:training_duration[0].index('/')]
training_end_point = training_duration[-1][:training_duration[-1].index('/')]
print('site: %s' % target_site_name)
print('Training for %s/%s to %s/%s' % (str(training_year[0]), training_duration[0],
                                       str(training_year[-1]), training_duration[-1]))
print('Testing for %s/%s to %s/%s' % (testing_year[0], testing_duration[0], testing_year[-1], testing_duration[-1]))
print('Target: %s' % target_kind)

# Set start time of data loading.
print('Loading data .. ')
start_time = time.time()
initial_time = time.time()

# Load training data, where: size(X_train) = (data_size, map_l, map_w, map_h), not sequentialized yet.
print('Preparing training dataset ..')
X_train = read_data_map(path=data_path, site=target_site, feature_selection=pollution_kind,
                        date_range=np.atleast_1d(training_year), beginning=training_duration[0],
                        finish=training_duration[-1], update=data_update)
X_train = missing_check(X_train)
Y_train = np.array(X_train)[:, center_i, center_j, pollution_kind.index(target_kind)-len(pollution_kind)]

# Load testing data, where: size(X_test) = (data_size, map_l, map_w, map_h), not sequentialized yet.
print('Preparing testing dataset ..')
X_test = read_data_map(path=data_path, site=target_site, feature_selection=pollution_kind,
                       date_range=np.atleast_1d(testing_year), beginning=testing_duration[0],
                       finish=testing_duration[-1], update=data_update)
X_test = missing_check(X_test)
Y_test = np.array(X_test)[:, center_i, center_j, pollution_kind.index(target_kind)-len(pollution_kind)]

# Set end time of data loading
final_time = time.time()
print('Reading data .. ok, ', end='')
time_spent_printer(start_time, final_time)

#
# ----- END: Data Loading ----------------------------------------------------------------------------------------------
#


#
# ----- START: Data Pre-processing -------------------------------------------------------------------------------------
#

# Normalize the dependent variable Y in the training dataset.
print('Normalize ..')
mean_y_train = np.mean(Y_train)
std_y_train = np.std(Y_train)
if not std_y_train:
    input("Denominator cannot be 0.")
Y_train = np.array([(y - mean_y_train) / std_y_train for y in Y_train])
print('mean_y_train: %f  std_y_train: %f' % (mean_y_train, std_y_train))
print('Feature processing ..')

# Construct sequential data.
print('Construct time series dataset ..')
start_time = time.time()
X_train = construct_time_map(X_train[:-1], train_seg_length)
X_test = construct_time_map(X_test[:-1], train_seg_length)
final_time = time.time()
time_spent_printer(start_time, final_time)

# Construct corresponding label.
Y_train = Y_train[train_seg_length:]
Y_train = topK_next_interval(Y_train, interval_hours, 1)
Y_test = Y_test[train_seg_length:]
Y_test = topK_next_interval(Y_test, interval_hours, 1)
Y_real = np.copy(Y_test)
Y_real = Y_real[:len(Y_test)]


# Compute the size of an epoch.
train_epoch_size = np.min([len(Y_train), len(X_train)])
test_epoch_size = np.min([len(Y_test), len(X_test)])
print('%d Training epoch size' % train_epoch_size)
print('%d Testing epoch size' % test_epoch_size)

# Make
X_train = X_train[:train_epoch_size]
Y_train = Y_train[:train_epoch_size]
Y_real = Y_real[:test_epoch_size]
X_test = X_test[:test_epoch_size]
Y_test = Y_test[:test_epoch_size]

# Delete testing data with missing values since testing data cannot be imputed.
i = 0
while i < len(Y_test):
    if not(Y_test[i] > -10000):  # check missing or not, if Y_test[i] is missing, then this command will return True
        Y_test = np.delete(Y_test, i, 0)
        Y_real = np.delete(Y_real, i, 0)
        X_test = np.delete(X_test, i, 0)
        i = -1
    i += 1

print('Delete invalid testing data, remain ', len(Y_test), 'test sequences')

#
# ----- END: Data Pre-processing ---------------------------------------------------------------------------------------
#


#
# ----- START: Data Partition ------------------------------------------------------------------------------------------
#

# Validation set
X_valid = X_train[-800:]
Y_valid = Y_train[-800:]

# Training set
X_train = X_train[:-800]
Y_train = Y_train[:-800]

print('take 800 data to validation set')

#
# ----- END: Data Partition --------------------------------------------------------------------------------------------
#





#
# ----- START: Model Definition ----------------------------------------------------------------------------------------
#

# Specify the path where model will be saved.
model_saved_path = ("rnn_%s_training_%s_m%s_to_%s_m%s_interval_%s_%s"
                    % (target_site_name, training_year[0], training_start_point, training_year[-1], training_end_point,
                       interval_hours, target_kind))
print(model_saved_path)

print('Build rnn model...')
start_time = time.time()

# for i in range(len(pollution_group)):
input_size = len(pollution_kind)+1 if 'WIND_DIREC' in pollution_kind else len(pollution_kind)  # feature 'WIND_DIREC' has two dimension
input_size = input_size + feature_kind_shift
input_shape = (train_seg_length, 5, 5, input_size)


# 5D tensor with shape: (samples_index, sequence_index, row, col, feature/channel)
model_input = Input(shape=input_shape, dtype='float32')

predict_map = Bidirectional(ConvLSTM2D(num_filters, kernel_size, padding='valid', activation='tanh',
                                       recurrent_activation='hard_sigmoid', use_bias=True, unit_forget_bias=True,
                                       kernel_regularizer=l2(regularizer), recurrent_regularizer=l2(regularizer),
                                       bias_regularizer=l2(regularizer), activity_regularizer=l2(regularizer),
                                       dropout=cnn_dropout, recurrent_dropout=r_dropout))(model_input)
predict_map = MaxPooling2D(pool_size=pool_size)(predict_map)
predict_map = Flatten()(predict_map)
# output layer
output_layer = BatchNormalization(beta_regularizer=None, epsilon=0.001, beta_initializer="zero", gamma_initializer="one",
                                  weights=None, gamma_regularizer=None, momentum=0.99, axis=-1)(predict_map)
output_layer = Dense(output_size, kernel_regularizer=l2(regularizer), bias_regularizer=l2(regularizer))(output_layer)


ConvLSTM_model = Model(inputs=model_input, outputs=output_layer)
ConvLSTM_model.compile(loss=keras.losses.mean_squared_error, optimizer='adam', metrics=['accuracy'])


final_time = time.time()
time_spent_printer(start_time, final_time)

#
# ----- END: Model Definition ------------------------------------------------------------------------------------------
#


#
# ----- START: Model Training ------------------------------------------------------------------------------------------
#

if is_training:
    print("Train...")
    start_time = time.time()

    ConvLSTM_model.fit(X_train, Y_train,
                       batch_size=batch_size,
                       epochs=epoch,
                       validation_data=(X_valid, Y_valid),
                       shuffle=True,
                       callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto'),
                                  ModelCheckpoint(folder + model_saved_path, monitor='val_loss', verbose=0,
                                                  save_best_only=False, save_weights_only=True, mode='auto', period=1)])

    # Potentially save weights
    ConvLSTM_model.save_weights(folder + model_saved_path, overwrite=True)

    final_time = time.time()
    time_spent_printer(start_time, final_time)
    print('model saved: ', model_saved_path)

else:
    print('loading model ..')
    ConvLSTM_model.load_weights(folder + model_saved_path)

#
# ----- END: Model Training --------------------------------------------------------------------------------------------
#


#
# ----- START: Model Testing -------------------------------------------------------------------------------------------
#

X_test_input = [X_test[:, j, :, :] for j in range(train_seg_length)]


ConvLSTM_pred = ConvLSTM_model.predict(X_test_input)
final_time = time.time()
time_spent_printer(start_time, final_time)

pred = mean_y_train + std_y_train * ConvLSTM_pred

print('rmse(rnn): %.5f' % (np.mean((np.atleast_2d(Y_test).T - pred)**2, 0)**0.5))

#
# ----- END: Model Testing ---------------------------------------------------------------------------------------------
#


'''
# 1st layer: TimeDistributed CNN
model.add(TimeDistributed(Conv2D(num_filters, kernel_size[0], kernel_size[1],
                                 border_mode="valid"), input_shape=[4, 1, 56, 14]))
model.add(TimeDistributed(Activation("relu")))

# 2nd layer: TimeDistributed CNN
model.add(TimeDistributed(Conv2D(num_filters, kernel_size[0], kernel_size[1])))
model.add(TimeDistributed(Activation("relu")))
model.add(TimeDistributed(MaxPooling2D(pool_size=pool_size)))
model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dropout(cnn_dropout)))

# 3rd layer: LSTM
model.add(LSTM(5, kernel_regularizer=l2(regularizer), recurrent_regularizer=l2(regularizer),
               bias_regularizer=l2(regularizer), activity_regularizer=l2(regularizer), recurrent_dropout=r_dropout))
model.add(Dropout(rnn_dropout))

# 4th layer: DNN
model.add(Dense(50))
model.add(Dense(num_classes, activation="softmax"))
#model.add(Activation("softmax"))
'''