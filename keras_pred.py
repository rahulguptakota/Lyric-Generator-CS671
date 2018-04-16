import keras
from keras.models import Sequential, model_from_json
from keras.layers import Activation,LSTM,Dense,Embedding
from keras.optimizers import Adam
from scipy.sparse import csr_matrix
import pickle, random

import re, json
import csv
import numpy as np
from nltk import word_tokenize

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
# print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


corpus=''
# for ix in range(len(data)):
#     corpus+=data[ix]
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
words_seq = words_seq[:150000]
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



generated=[]
# start_index=random.randint(0,len(words_seq)-maxlen-1)
start_index=0
sent=words_seq[start_index:start_index+maxlen]
generated+=sent
for i in range(100):
    x_sample=generated[i:i+maxlen]
    x=np.zeros((1,maxlen))
    print(x_sample)
    for j in range(maxlen):
        x[0,j]=word_ix[x_sample[j]]
    probs=loaded_model.predict(x)
    print(i, probs)
    probs=np.reshape(probs,probs.shape[1])
    ix=np.random.choice(range(vocab_size),p=probs.ravel())
    print(i, ix)
    generated+=[ix_word[ix]]

print(generated)