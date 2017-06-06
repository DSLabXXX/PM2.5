# _*_ coding: utf-8 _*_

import numpy as np

from reader import read_data_sets, concatenate_time_steps, construct_time_steps
from missing_value_processer import missing_check
from feature_processor import data_coordinate_angle


class Gen_Data_loader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.token_stream = []
        self.X = []  # input of the rnn of the first layer for sequence input to an embedding vector

    def load_data(self, data_file, site_list, target_site, target_kind, training_year, training_duration,
                  pollution_kind, SEQ_LENGTH_1, SEQ_LENGTH_2, data_update=False):
        print('Reading data .. ')
        X = read_data_sets(sites=site_list+[target_site], date_range=np.atleast_1d(training_year),
                           beginning=training_duration[0], finish=training_duration[-1],
                           feature_selection=pollution_kind, update=data_update)
        X = missing_check(X)
        Y = np.array(X)[:, -len(pollution_kind):]
        Y = Y[:, pollution_kind.index(target_kind)]
        SeqY = []
        for y in range(len(Y)):
            if (y + (SEQ_LENGTH_2 - 1)) < len(Y):
                Seqy = []
                for time_step in range(SEQ_LENGTH_2):
                    Seqy.append(Y[y+time_step])
                SeqY.append(Seqy)
                del Seqy
            else:
                break
        X = np.array(X)[:, :-len(pollution_kind)]

        # feature process
        if 'WIND_DIREC' in pollution_kind:
            index_of_kind = pollution_kind.index('WIND_DIREC')
            length_of_kind_list = len(pollution_kind)
            len_of_sites_list = len(site_list)
            X = X.tolist()
            for i in range(len(X)):
                for j in range(len_of_sites_list):
                    specific_index = index_of_kind + j * length_of_kind_list
                    coordin = data_coordinate_angle(X[i].pop(specific_index + j))
                    X[i].insert(specific_index + j, coordin[1])
                    X[i].insert(specific_index + j, coordin[0])
            X = np.array(X)

        X = construct_time_steps(X[:-1], SEQ_LENGTH_1)

        if SEQ_LENGTH_1 < SEQ_LENGTH_2:
            self.X = X[0:len(SeqY)]
        elif SEQ_LENGTH_1 > SEQ_LENGTH_2:
            SeqY = SeqY[:len(X)]

        with open(data_file, 'w') as f:
            for line in SeqY:
                for elem_no in range(len(line)):
                    f.write(str(line[elem_no]))
                    if elem_no < (len(line) - 1):
                        f.write(' ')
                f.write('\n')

    def create_batches(self, data_file, SEQ_LEN):
        self.token_stream = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [float(x) for x in line]
                if len(parse_line) == SEQ_LEN:
                    self.token_stream.append(parse_line)

        num_batch = int(len(self.X) / self.batch_size)
        self.X = self.X[:num_batch * self.batch_size]
        self.X = np.split(np.array(self.X), num_batch, 0)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.sequence_batch = np.reshape(self.sequence_batch, (self.num_batch, self.batch_size, SEQ_LEN, 1))
        self.pointer = 0

    def next_batch(self):
        ret_x = self.X[self.pointer]
        ret_y = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret_x, ret_y

    def reset_pointer(self):
        self.pointer = 0


class Dis_dataloader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])

    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                positive_examples.append(parse_line)
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == 20:
                    negative_examples.append(parse_line)
        self.sentences = np.array(positive_examples + negative_examples)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0
