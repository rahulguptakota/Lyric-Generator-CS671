import re, nltk
import random
import operator
from nltk import pos_tag, word_tokenize
import random, re, sys, pronouncing
# reload(sys)
# sys.setdefaultencoding('utf8')

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

def markov_next(curr, probDict,posDict):
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
        # print ("yo")
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
  pos_lyrics("oldsong.txt")
  input_file = open("oldsong.pos", "r")
  text = input_file.read()
  line = text.split()
  rapProbDict = addToDict('oldsong.txt', rapFreqDict)

  #calcuate prior probabilites in "oldsong.pos"
  probabilities(line)
  
  #add each word in lyrics to a list and return it
  the_song,pos_dict = lyrics(rapProbDict)
  print_lyrics(the_song,pos_dict) #you know what to do

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
  while total_words != 250:
    pick = []
    #based on the first word, come up with the other words!
    
    #determine POS for current word
    current_word = lyrics[total_words]
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
        count += 1
        for words in pos_dict[word]:
          #randomly pick some common Drake word
          #add_word = random.choice(pos_dict[word].keys())
          # print (pos_dict[word].keys())
          add_word = markov_next(current_word, probDict,pos_dict[word])
          lyrics.append(add_word)
          if (count ==1):
            count = 0
            break
    total_words += 1
  return lyrics,pos_dict

def print_lyrics(lyrics,pos_dict):
  output = open("newsong1.txt", "w")
  count = 0
  #hardcode title to be first three words
  title = lyrics[:3]
  #hardcode chorus to be next 50 words
  chorus = lyrics[3:53]
  #hardcode verses to be remaining
  verses = lyrics[53:]
  output.write("TITLE:")
  for words in title:
    output.write(words + " ")
  output.write("\n\n")
  print_chorus(chorus, output,pos_dict)
  output.write("\n\n")
  count1=0
  for words in lyrics:
    # output.write(words + " ")
    count += 1
    if count % 10 ==0:
      if count1==0:
        prev_word = lyrics[count-1]
        # print ("prev_word",prev_word)
        # idx+=random_idx+1
        rhymed_word = pronouncing.rhymes(prev_word)
        # print ("rhymed_word",rhymed_word)
        while not rhymed_word:
          tag_word = nltk.pos_tag([prev_word])[0][1]
          prev_word = random.choice(pos_dict[tag_word].keys())
          # ix_word = generated[idx]
          rhymed_word = pronouncing.rhymes(prev_word)
          # idx+=1
        words1=prev_word

        # print ("ix_word",ix_word)
      if count1:
        added_word=0
        # print ("words",words)
        # print ("prev_word",prev_word)
        tag_word = nltk.pos_tag([prev_word])[0][1]
        # print ("tag_word",tag_word)
        rhymed_word = pronouncing.rhymes(prev_word)
        # print ("rhymed_word",rhymed_word)
        for word in rhymed_word:
            if (nltk.pos_tag([word])[0][1])==tag_word:
                added_word=1
                words1=word
                break
        # print ("sent",sent)
        # print ("added_word",added_word)
        # if added_word==0:
        #     if not rhymed_word:
        #         sent+=[generated[random_idx+1]]
        #         idx+=random_idx+2
        #     else:
        #         sent+=[rhymed_word[0]]
        #         idx+=random_idx+1
        # else:
        #     idx+=random_idx+1
        count1=0
      else:
          # prev_word=ix_word
          # sent+=[ix_word]
          count1=1
      output.write(words1 + " ")
      # print ("words1",words1)
      output.write("\n")
      # print ("word",words)
      # print ("count",count)
    elif count % 83 == 0:
      output.write("\n\n")
      # print_chorus(chorus, output,pos_dict)
      # output.write("\n\n")
      count1=0
    else:
      output.write(words + " ")
  output.close()

def print_chorus(words_in_chorus, output,pos_dict):
  count = 0
  count1 = 0
  output.write("CHORUS:")

  for words in words_in_chorus:
    # output.write(words + " ")
    count += 1
    if count % 10 ==0:
      if count1==0:
        prev_word = words_in_chorus[count-1]
        # print ("prev_word",prev_word)
        # idx+=random_idx+1
        rhymed_word = pronouncing.rhymes(prev_word)
        # print ("rhymed_word",rhymed_word)
        while not rhymed_word:
          tag_word = nltk.pos_tag([prev_word])[0][1]
          prev_word = random.choice(pos_dict[tag_word].keys())
          # ix_word = generated[idx]
          rhymed_word = pronouncing.rhymes(prev_word)
          # idx+=1
        words1=prev_word

        # print ("ix_word",ix_word)
      if count1:
        added_word=0
        # print ("words",words)
        # print ("prev_word",prev_word)
        tag_word = nltk.pos_tag([prev_word])[0][1]
        # print ("tag_word",tag_word)
        rhymed_word = pronouncing.rhymes(prev_word)
        # print ("rhymed_word",rhymed_word)
        for word in rhymed_word:
            if (nltk.pos_tag([word])[0][1])==tag_word:
                added_word=1
                words1=word
                break
        # print ("sent",sent)
        # print ("added_word",added_word)
        # if added_word==0:
        #     if not rhymed_word:
        #         sent+=[generated[random_idx+1]]
        #         idx+=random_idx+2
        #     else:
        #         sent+=[rhymed_word[0]]
        #         idx+=random_idx+1
        # else:
        #     idx+=random_idx+1
        count1=0
      else:
          # prev_word=ix_word
          # sent+=[ix_word]
          count1=1
      output.write(words1 + " ")
      # print ("words1",words1)
      output.write("\n")
    else:
      output.write(words + " ")
      # print ("word",words)
      # print ("count",count)
      

def split(s, n):
  return s[n:]

if __name__ == "__main__":
      main()
