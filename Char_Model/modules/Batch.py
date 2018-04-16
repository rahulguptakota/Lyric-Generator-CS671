#!/usr/bin/env python
__author__ = 'Tony Beltramelli www.tonybeltramelli.com - 20/08/2016'

import codecs
import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
# from modules.Vocabulary import *
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Batch:
    dataset_full_passes = 0

    def __init__(self, data_file_name, vocabulary_file_path, batch_size, sequence_length):
        self.data_file = codecs.open(data_file_name, 'r', 'utf_8')

        # self.vocabulary = Vocabulary()
        # self.vocabulary.retrieve(vocabulary_file_path)
        self.vocabulary = KeyedVectors.load_word2vec_format("glove.6B.50d.word2vec")
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def get_wordvec(self, word):
        try:
            self.vocabulary[word]
            return self.vocabulary[word]
        except:
            return np.zeros(50)

    def get_next_batch(self):
        string_len = self.batch_size * self.sequence_length + self.batch_size
        current_batch = self.data_file.read(string_len)
        batch_vector = []
        label_vector = []

        if len(current_batch) < string_len:
            while len(current_batch) < string_len:
                current_batch += u' '
            self.data_file.seek(0)
            self.dataset_full_passes += 1
            print("Pass {} done".format(self.dataset_full_passes))
        print("batch size: ", self.batch_size)
        for i in np.arange(0, string_len, self.sequence_length + 1):
            print("sequences limits: ", i, i + self.sequence_length)
            sequence = current_batch[i:i + self.sequence_length]
            label = current_batch[i + self.sequence_length:i + self.sequence_length + 1]
            sequences_vector = []
        
            for char in sequence:
                sequences_vector.append(self.get_wordvec(char))
            batch_vector.append(sequences_vector)
            label_vector.append(self.get_wordvec(char))
        # print("printing batch vectors: ", np.asarray(batch_vector), np.asarray(label_vector))
        return np.asarray(batch_vector), np.asarray(label_vector)

    def clean(self):
        self.data_file.close()
