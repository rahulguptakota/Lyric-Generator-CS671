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
from keras.models import model_from_json



def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    Glove_model = {}
    Glove_words = []
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        Glove_words.insert(0,word)
        embedding = np.array([float(val) for val in splitLine[1:]])
        Glove_model[word] = embedding
    print("Done.",len(Glove_model)," words loaded!")
    return Glove_model, Glove_words

glove_model, glove_words = loadGloveModel("glove.6B.50d.txt")



json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
# print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','top_k_categorical_accuracy'])

####### loading/storing corpus
# fp = open("corpus.p", "wb")
# pickle.dump(corpus, fp)
fp = open("corpus.p", "rb")
corpus = pickle.load(fp)
fp.close()

corpus = re.sub(r"\([^\n]*\)", " ", corpus)
corpus = re.sub(r"\([^\n]*\)", " ", corpus)
corpus = re.sub(r"\n+", " ttttttttttt ", corpus)
corpus=corpus.lower()

print("corpus constructed")
# print(corpus)
# exit()
words_seq = word_tokenize(corpus)
words_seq = [token.lower() for token in words_seq]
prev_word = ''
new_seq = []
for i in range(len(words_seq)):
    if words_seq[i]=="ttttttttttt":
        new_seq.append("\n")
    elif words_seq[i] != prev_word:
        new_seq.append(words_seq[i])
        prev_word = words_seq[i]

print("length of words_seq: ", len(words_seq))
words_seq = new_seq[:300000]

print("length of words_seq: ", len(words_seq))
words_seq = words_seq[:300000]

print(words_seq)

# vocab=list(set(words_seq))
# fp = open("vocab.p", "wb")
# pickle.dump(vocab, fp)
fp = open("vocab.p", "rb")
vocab = pickle.load(fp)
fp.close()
print(len(vocab))
word_ix={c:i for i,c in enumerate(vocab)}
ix_word={i:c for i,c in enumerate(vocab)}
# print(ix_word)
maxlen=3
batch_size = 128
vocab_size=len(vocab)

sentences=[]
next_word=[]
for i in range(len(words_seq)-maxlen-1):
    sentences.append(words_seq[i:i+maxlen])
    next_word.append(words_seq[i+maxlen])

print(len(sentences),maxlen,vocab_size)
previous_word = ""
def find_nearest(vec, method='cosine'):
    if method == 'cosine':
        nearest_word = ""
        max_val = -1
        list_vec = []
        for word in vocab:
            try:
                val = np.dot(vec, glove_model[word])
                list_vec.insert(0,[val, word])
                if max_val < val:
                    max_val = val
                    nearest_word = word
            except Exception as e:
                pass
        sorted_list_vec = sorted(list_vec, key=lambda x: x[0])
        # if previous_word == sorted_list_vec[0][1]:
        #     return sorted_list_vec[1][0],sorted_list_vec[1][1]
        # else:
        #     return sorted_list_vec[0][0],sorted_list_vec[0][1]
        len_gen = len(generated)
        for y in range(len(sorted_list_vec)):   
            flag = 0         
            for x in generated:
                if x == sorted_list_vec[y][1]:
                    break
                flag+=1
            if flag == len_gen:
                return sorted_list_vec[y][0],sorted_list_vec[y][1] 
    else:
        raise Exception('{} is not an excepted method parameter'.format(method))



generated=[]
start_index=random.randint(0,len(words_seq)-maxlen-1)

# start_index=1

sent=words_seq[start_index:start_index+maxlen]
generated+=sent
for i in range(100):
    x_sample=generated[i:i+maxlen]
    x=np.zeros((1,maxlen*50))
    print(x_sample)
    # for j in range(maxlen):
    #     x[0,j]=word_ix[x_sample[j]]

    for ix in range(0):
        # choose random index in features
        try:
            for iy in range(maxlen):
                # print(iy)
                x[0][iy*50:(iy+1)*50] = list(glove_model[x_sample[iy]])
        except:
            ix -= 1

    probs=loaded_model.predict(x)
    # print(i, probs)
    # probs=np.reshape(probs,probs.shape[1])
    prob, word = find_nearest(probs)
    previous_word = word

    # ix=list(probs).index(max(probs))
    # ix=np.random.choice(range(vocab_size),p=probs.ravel())
    # print(i, ix)
    generated+=[word]

print(generated)
