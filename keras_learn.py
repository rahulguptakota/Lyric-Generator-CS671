import keras
from keras.models import Sequential
from keras.layers import Activation,LSTM,Dense
from keras.optimizers import Adam
import pickle

import re, json
import csv
import numpy as np
from nltk import word_tokenize

df = []
with open('songdata.csv', 'r') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',')
	i = 0
	for row in spamreader:
		df.append(row[3][:100])
		i+=1
		# print("-------------------------------------------------------")
		# if i > 5:
		# 	break

data=np.array(df)
print(data)

corpus=''
# for ix in range(len(data)):
#     corpus+=data[ix]
# fp = open("corpus.p", "wb")
# pickle.dump(corpus, fp)
fp = open("corpus.p", "rb")
corpus = pickle.load(fp)
fp.close()

corpus = re.sub(r"\n+", " \n ", corpus)

print("corpus constructed")
# print(corpus)
words_seq = corpus.split(' ')
vocab=list(set(words_seq))
fp = open("vocab.p", "wb")
pickle.dump(vocab, fp)
# fp = open("vocab.p", "rb")
# vocab = pickle.load(fp)
fp.close()
print(len(vocab))
word_ix={c:i for i,c in enumerate(vocab)}
ix_word={i:c for i,c in enumerate(vocab)}

maxlen=5
vocab_size=len(vocab)

sentences=[]
next_word=[]
for i in range(len(words_seq)-maxlen-1):
    sentences.append(words_seq[i:i+maxlen])
    next_word.append(words_seq[i+maxlen])

print(len(sentences),maxlen,vocab_size)

X=np.zeros((len(sentences),maxlen,vocab_size))
y=np.zeros((len(sentences),vocab_size))
for ix in range(len(sentences)):
    y[ix,word_ix[next_word[ix]]]=1
    for iy in range(maxlen):
        X[ix,iy,word_ix[sentences[ix][iy]]]=1

model=Sequential()
model.add(LSTM(128,input_shape=(maxlen,vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
model.summary()
model.compile(optimizer=Adam(lr=0.01),loss='categorical_crossentropy')

model.fit(X,y,epochs=5,batch_size=128)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

import random
generated=''
start_index=random.randint(0,len(words_seq)-maxlen-1)
sent=words_seq[start_index:start_index+maxlen]
generated+=sent
for i in range(100):
    x_sample=generated[i:i+maxlen]
    x=np.zeros((1,maxlen,vocab_size))
    for j in range(maxlen):
        x[0,j,word_ix[x_sample[j]]]=1
    probs=model.predict(x)
    probs=np.reshape(probs,probs.shape[1])
    ix=np.random.choice(range(vocab_size),p=probs.ravel())
    generated+=ix_word[ix]

print(generated)

