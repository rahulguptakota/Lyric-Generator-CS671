import csv
import json
import pickle
import random
import re

import keras
import numpy as np
from keras.layers import LSTM, Activation, Dense, Embedding, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from nltk import word_tokenize
from scipy.sparse import csr_matrix

df = []
with open('songdata.csv', 'r') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',')
	i = 0
	for row in spamreader:
		df.append(row[3])
		i+=1
		# print("-------------------------------------------------------")
		# if i > 5:
		# 	break
data=np.array(df)
print("Length of data: ", len(data))

corpus=''
for ix in range(15000):
    corpus+=" "
    corpus+=data[ix]
    # print(ix)

# print(corpus)

fp = open("corpus.p", "wb")
pickle.dump(corpus, fp)
# fp = open("corpus.p", "rb")
# corpus = pickle.load(fp)
fp.close()
# corpus=corpus[]
corpus = re.sub(r"\([^\n]*\)", " ", corpus)
corpus = re.sub(r"\([^\n]*\)", " ", corpus)
corpus = re.sub(r"\n+", " ", corpus)

print("corpus constructed")
# print(corpus)
# exit()
words_seq = word_tokenize(corpus)
words_seq = [token.lower() for token in words_seq]
# for i in range(len(words_seq)):
#     if words_seq[i]=="ttttttttttt":
#         words_seq[i] = ""

print("length of words_seq: ", len(words_seq))
words_seq = words_seq[:200000]
print(words_seq)

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


def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    Glove_model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        Glove_model[word] = embedding
    print("Done.",len(Glove_model)," words loaded!")
    return Glove_model

glove_model = loadGloveModel("glove.6B.50d.txt")

# glove_model[word]

def generator(batch_size):
    global word_ix, sentences, vocab_size, maxlen, next_word
    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, maxlen*50))
    batch_labels = np.zeros((batch_size, 50))

    # batch_labels = np.zeros((batch_size,vocab_size), dtype=np.bool)
    while True:
        for ix in range(batch_size):
            # choose random index in features
            try:
                index = random.choice(range(len(sentences)))
                batch_labels[ix] = glove_model[next_word[index]]
                for iy in range(maxlen):
                    # print(iy)
                    batch_features[ix][iy*50:(iy+1)*50] = list(glove_model[sentences[index][iy]])
                # batch_labels[ix,word_ix[next_word[index]]] = 1
                # for iy in range(maxlen):
                # batch_features[ix,iy]=word_ix[sentences[index][iy]]
            except:
                ix -= 1
        yield batch_features, batch_labels

max_features = maxlen*50
# model=Sequential()
# model.add(Embedding(max_features, input_length=maxlen*50 ,output_dim=256))
# model.add(LSTM(128))
# model.add(Dense(50))
# model.add(Activation('softmax'))
# model.summary()



# # def custom_pred(y_true, y_pred):
# #     # dot_prod = 0
# #     # for i in range(y_true.size()):
# #     #     dot_prod += y_true[i]*y_pred[i]
# #     import keras.backend as K
# #     C = K.sum(y_true * y_pred,axis=-1,keepdims=True)
# #     # val = K.cast_to_floatx(C)
# #     val = K.eval(C)
# #     # print(int(C))
# #     # print(C.size())
# #     # print(C.shape())
# #     print(val)
# #     if val > 0.4:
# #         return 1
# #     else:
# #         return 0


# model.compile(optimizer=Adam(lr=0.1),loss='categorical_crossentropy',metrics=['accuracy'])

# model.fit_generator(generator(batch_size), samples_per_epoch=len(sentences)/batch_size, nb_epoch=10)

model = Sequential()
model.add(Dense(100, activation="relu", kernel_initializer="uniform", input_dim=maxlen*50))
model.add(Dense(50, activation="relu", kernel_initializer="uniform"))
model.add(Dropout(0.5))
model.add(Dense(50, activation='softmax'))

# sgd = SGD(lr=0.01)
# model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.compile(optimizer=Adam(lr=0.1),loss='mean_squared_error',metrics=['accuracy'])

# model.fit(np.array(X_train), np.array(y_train), epochs=50, batch_size=128)
model.fit_generator(generator(batch_size), samples_per_epoch=len(sentences)/batch_size, nb_epoch=10)



# for i in range(1000):
#     model.fit(X,y,epochs=5,batch_size=128)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
