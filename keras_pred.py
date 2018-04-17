import keras
from keras.models import Sequential
from keras.layers import Activation,LSTM,Dense,Embedding
from keras.optimizers import Adam
from scipy.sparse import csr_matrix
import pickle, random

import re, json
import csv
import numpy as np
from nltk import word_tokenize
from keras.models import model_from_json

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
# print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','top_k_categorical_accuracy'])
