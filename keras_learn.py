import csv
import json
import pickle
import random
import re

import keras
import numpy as np
from keras.layers import LSTM, Activation, Dense, Embedding
from keras.models import Sequential
from keras.optimizers import Adam
from nltk import word_tokenize
from scipy.sparse import csr_matrix

df = []
# with open('songdata.csv', 'r') as csvfile:
# 	spamreader = csv.reader(csvfile, delimiter=',')
# 	i = 0
# 	for row in spamreader:
# 		df.append(row[3])
# 		i+=1
# 		# print("-------------------------------------------------------")
# 		# if i > 5:
# 		# 	break
# data=np.array(df)
# print(len(data))

corpus=''
# for ix in range(1000):
#     corpus+=data[ix]
#     # print(ix)
# fp = open("corpus.p", "wb")
# pickle.dump(corpus, fp)
fp = open("corpus.p", "rb")
corpus = pickle.load(fp)
fp.close()
# corpus=corpus[]
corpus = re.sub(r"\n+", " \n ", corpus)

print("corpus constructed")
# print(corpus)
words_seq = corpus.split(' ')
print("length of words_seq: ", len(words_seq))
words_seq = words_seq[:300000]
vocab=list(set(words_seq))
fp = open("vocab.p", "wb")
pickle.dump(vocab, fp)
# fp = open("vocab.p", "rb")
# vocab = pickle.load(fp)
fp.close()
print(len(vocab))
word_ix={c:i for i,c in enumerate(vocab)}
ix_word={i:c for i,c in enumerate(vocab)}
# print(ix_word)
maxlen=5
batch_size = 128
vocab_size=len(vocab)

sentences=[]
next_word=[]
for i in range(len(words_seq)-maxlen-1):
    sentences.append(words_seq[i:i+maxlen])
    next_word.append(words_seq[i+maxlen])

print(len(sentences),maxlen,vocab_size)

# def get_Xy(batch_size, i):
#     global word_ix, sentences, vocab_size, maxlen, next_word
#     X=np.zeros((batch_size,maxlen))
#     y=np.zeros((batch_size,vocab_size), dtype=np.bool)
#     for ix in range(i*batch_size, (i+1)*batch_size):
#         y[ix,word_ix[next_word[ix]]]=1
#         for iy in range(maxlen):
#             X[ix,iy]=word_ix[sentences[ix][iy]]
#     return X,y

def generator(batch_size):
    global word_ix, sentences, vocab_size, maxlen, next_word
    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, maxlen))
    batch_labels = np.zeros((batch_size,vocab_size), dtype=np.bool)
    while True:
        for ix in range(batch_size):
            # choose random index in features
            index = random.choice(range(len(sentences)))
            batch_labels[ix,word_ix[next_word[index]]] = 1
            for iy in range(maxlen):
                batch_features[ix,iy]=word_ix[sentences[index][iy]]
        yield batch_features, batch_labels

max_features = vocab_size
model=Sequential()
model.add(Embedding(max_features, input_length=maxlen ,output_dim=256))
model.add(LSTM(128))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
model.summary()
model.compile(optimizer=Adam(lr=0.01),loss='categorical_crossentropy',metrics=['accuracy','top_k_categorical_accuracy'])

model.fit_generator(generator(batch_size), samples_per_epoch=len(sentences)/batch_size, nb_epoch=10)

# for i in range(1000):
#     model.fit(X,y,epochs=5,batch_size=128)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")