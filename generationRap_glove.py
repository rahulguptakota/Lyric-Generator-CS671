import re
import random
import operator
from nltk import pos_tag, word_tokenize
import random, re, sys, pronouncing


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


reload(sys)
sys.setdefaultencoding('utf8')

# freqDict is a dict of dict containing frequencies
def addToDict(fileName, freqDict):
  f = open(fileName, 'r')
  # words = re.sub("\n", " \n", f.read()).lower().split(' ')
  words = re.sub("\n", " \n", f.read()).split(' ')
  # text = f.read()
  # words = text.split()

  # count frequencies curr -> succ
  for curr, succ in zip(words[1:], words[:-1]):
    # check if curr is already in the dict of dicts
    if curr not in freqDict:
      freqDict[curr] = {succ: 1}
    else:
      # check if the dict associated with curr already has succ
      if succ not in freqDict[curr]:
        freqDict[curr][succ] = 1;
      else:
        freqDict[curr][succ] += 1;

  # compute percentages
  probDict = {}
  for curr, currDict in freqDict.items():
    probDict[curr] = {}
    currTotal = sum(currDict.values())
    for succ in currDict:
      probDict[curr][succ] = currDict[succ] / currTotal
  return probDict


z = 0
def markov_next(curr, probDict,posDict):
  global z
  # random.choice(pos_dict[word].keys())
  # random.choice(list(probDict.keys()))
  if curr not in probDict:
    return random.choice(posDict.keys())
  else: 
    succProbs = probDict[curr]
    randProb = random.random()
    currProb = 0.0
    maxima=0.0
    name=""
    for succ in succProbs:
      currProb += succProbs[succ]
      # randProb <= currProb and
      if succ in posDict.keys():
        print ("yo",z)
        z+=1
        return succ
    return random.choice(posDict.keys())

prior_probs_dict = {} #number of occurrences for each prior prior probability pair
total_pos = {} #total number of occurrences for each pos
pos_dict = {} #dict of dicts that contains each part of speech, the words that make up those parts of speech, and number of each occurrence for each word
word_dict = {} #dictionary of words and occurrences of each corresponding pos
subsequent_dict = {} #dictionary of probabilities. showcases the likelihood of a given pos following another pos
count = 0
pos_count = 0
pos_count_2 = 0

def main():
  #use nltk to determine parts of speech for old songs, write to "oldsong.pos"
  rapFreqDict = {}
  #rapProbDict = addToDict('lyrics3.txt', rapFreqDict)
  # rapProbDict = addToDict('oldsong.txt', rapFreqDict)
  pos_lyrics("poemdata.txt")
  input_file = open("oldsong.pos", "r")
  text = input_file.read()
  line = text.split()
  rapProbDict = addToDict('poemdata.txt', rapFreqDict)

  #calcuate prior probabilites in "oldsong.pos"
  probabilities(line)
  
  #add each word in lyrics to a list and return it
  the_song = lyrics(rapProbDict)
  print_lyrics(the_song) #you know what to do

def pos_lyrics(input_file):
  f_read = open(input_file, "r")
  f_write = open("oldsong.pos", "w")
  text = f_read.read()
  list_of_pos = pos_tag(word_tokenize(text))
  for word in list_of_pos:
      f_write.write(word[0].lower() + "  " + word[1] + "\n")
  f_read.close()
  f_write.close()

def probabilities(line):
  pos_count = 0
  pos_count_2 = 0
  hmm_count = 0
  count = 0
  #find prior probabilities and add to a dictionary
  for word in range(int(len(line)/2-1)):
    word = line[count + 1] + " "
    following_word = line[count + 3] + " "
    word_combo = word + "-> " + following_word
    # check if word_combo exists in prior_probs_dict dictionary
    if word_combo in prior_probs_dict:
      #add to count
      prior_probs_dict[word_combo] += 1
    else:
      #create key and add to count
      prior_probs_dict[word_combo] = 1
    count += 2
    if len(line) == count:
      count = 0
      break

  #find total number of occurrences for each pos
  for word in range(int(len(line)/2-1)):
    #total up each pos in total_pos dictionary
    pos = line[pos_count +1] + " "
    if pos in total_pos:
      total_pos[pos] += 1
    else:
      total_pos[pos] = 1
    pos_count += 2

  for pos in total_pos:
    subsequent_dict[pos] = {}
    for word_combo in prior_probs_dict:
        if word_combo.startswith(pos):
          #probability is the number of word_combos/total occurrences of pos
          occurrences_of_word_combo = prior_probs_dict[word_combo]
          occurrences_of_pos = total_pos[pos]
          probability = (occurrences_of_word_combo)/float(occurrences_of_pos)
          subsequent_dict[pos][word_combo] = probability

  #have a dictionary of dictionaries that contains each part of speech, the words that make up those parts of speech, and number of each occurrence for each word
  for word in range(len(line)):
    word = line[pos_count_2]
    pos = line[pos_count_2 +1]
    #check if pos exists in dictionary
    #POS DICT-------------------------------
    if pos in pos_dict:
        #check to see if word is already in nested dict. if not, add
        if word in pos_dict[pos]:
          pos_dict[pos][word] += 1
        else:
          pos_dict[pos][word] = 1
    else:
        #add that pos to the dict, add add word to nested dict
        pos_dict[pos] = {}
        pos_dict[pos][word] = 1
    #WORD DICT-------------------------------
    if word in word_dict:
        #check to see if word is already in nested dict. if not, add
        if pos in word_dict[word]:
          word_dict[word][pos] += 1
        else:
          word_dict[word][pos] = 1
    else:
        #add that pos to the dict, add add word to nested dict
        word_dict[word] = {}
        word_dict[word][pos] = 1
    pos_count_2 += 2
    if len(line) == pos_count_2:
      pos_count_2 = 0
      break






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

def find_nearest(vec, prev_word, method='cosine'):
    if method == 'cosine':
        nearest_word = ""
        max_val = -1
        list_vec = []
        rhymed_words = pronouncing.rhymes(prev_word)
        for word in rhymed_words:
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
        len_gen = len(rhymed_words)
        for y in range(len(sorted_list_vec)):   
            flag = 0         
            for x in rhymed_words:
                if x == sorted_list_vec[y][1]:
                    break
                flag+=1
            if flag == len_gen:
                return sorted_list_vec[y][0],sorted_list_vec[y][1] 
    else:
        raise Exception('{} is not an excepted method parameter'.format(method))



def lyrics(probDict):
  lyrics = []
  total_words = 0 #no more than 250 words/song
  count = 0
  #hardcode it so that first word is a noun, yolo
  startWord = raw_input("What do you want to start your rap with?\n > ")
  lyrics.append(startWord)
  #print("Alright, here's your rap:")
  # for word in pos_dict:
  #   if word == "NN":
  #     count += 1
  #     for words in pos_dict[word]:
  #       #randomly pick some common Drake noun
  #       #add_word = markov_next(rap[-1], probDict,pos_dict[word])
  #       add_word = random.choice(pos_dict[word].keys())
  #       lyrics.append(add_word)
  #       if (count ==1):
  #         count = 0
  #         break
  next_newline = 7
  line =1
  while total_words != 500:
    pick = []
    #based on the first word, come up with the other words!
    
    #determine POS for current word
    current_word = lyrics[total_words]
    if current_word == "\n":
      current_word = lyrics[total_words-1]
    pos_of_current_word = max(word_dict[current_word].iteritems(), key=operator.itemgetter(1))[0] + " "
    
    #determine top two possible POS for subsequent word
    pos_options = dict(sorted(subsequent_dict[pos_of_current_word].iteritems(), key=operator.itemgetter(1), reverse=True)[:2])
    #pick one at random
    full_pos = random.choice(pos_options.keys())
    length_of_old_pos = len(pos_of_current_word) + 3
    pos_of_next_word = split(full_pos, length_of_old_pos)
    
    #generate a random word from the list of words for that given part of speech
    for word in pos_dict:
      if (word + " ") == pos_of_next_word:
        
        for words in pos_dict[word]:
          #randomly pick some common Drake word
          #add_word = random.choice(pos_dict[word].keys())
          # print (pos_dict[word].keys())
          

          add_word = markov_next(current_word, probDict,pos_dict[word])
          lyrics.append(add_word)
          total_words += 1
          count += 1
          if count == next_newline:
              lyrics.append("\n")
              line += 1
              next_newline += random.randint(6, 12)
              if line % 2:
                previous_word_odd = add_word 
              else:
                previous_word_even = add_word

          try:
            if line>=3:
            #########################################################################
              maxlen = 3
              x_sample=lyrics[len(lyrics)-3:len(lyrics)]
              x=np.zeros((1,maxlen*50))
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
              if line % 2:
                prob, word = find_nearest(probs, previous_word_odd)
              else:
                prob, word = find_nearest(probs, previous_word_even)

              previous_word = word
              lyrics.append(word)
              count += 1
              total_words += 1

          except:
            pass
          if count == next_newline:
            lyrics.append("\n")
            line += 1
            next_newline += random.randint(6, 12)
            if line % 2:
              previous_word_odd = word 
            else:
              previous_word_even = word
          #########################################################################
    # total_words += 1
  return lyrics
def print_lyrics(lyrics):
  print(lyrics)
# def print_lyrics(lyrics):
#   output = open("newsong1.txt", "w")
#   count = 0
#   #hardcode title to be first three words
#   title = lyrics[:3]
#   #hardcode chorus to be next 50 words
#   chorus = lyrics[3:53]
#   #hardcode verses to be remaining
#   verses = lyrics[53:]
#   output.write("TITLE:")
#   for words in title:
#     output.write(words + " ")
#   output.write("\n\n")
#   print_chorus(chorus, output)
#   output.write("\n\n")
#   for words in lyrics:
#     output.write(words + " ")
#     count += 1
#     if count % 10 ==0:
#       output.write("\n")
#     elif count % 83 == 0:
#       output.write("\n\n")
#       print_chorus(chorus, output)
#       output.write("\n\n")
#   output.close()

# def print_chorus(words_in_chorus, output):
#   count = 0
#   output.write("CHORUS:")
#   for words in words_in_chorus:
#     output.write(words + " ")
#     count += 1
#     if count % 10 ==0:
#       output.write("\n")

def split(s, n):
  return s[n:]

if __name__ == "__main__":
      main()
