# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from __future__ import division
import pandas as pd
import math
import copy
import numpy as np
import pickle
import sys

path = "/home/stufs1/nitigupta/628/data/train_v2_1719648.txt" #path to the tagged file
test_data_path = "/home/stufs1/nitigupta/628/data/test_v2_tagged_full.txt"
bi_word_path = "/home/stufs1/nitigupta/628/pickled_ds/bi_word.txt" #bigram word to word dict
bi_arr_path = "/home/stufs1/nitigupta/628/pickled_ds/bi_arr.txt" #bigram pos to pos arr
tri_arr_path = "/home/stufs1/nitigupta/628/pickled_ds/tri_arr.txt" #trigram pos to pos arr
pos_prob_path = "/home/stufs1/nitigupta/628/pickled_ds/pos_prob_path.txt" #pos prob dict
word_pos_path = "/home/stufs1/nitigupta/628/pickled_ds/word_pos_path.txt" #prev word to cur pos prob dict
pos_word_dict_path = "/home/stufs1/nitigupta/628/pickled_ds/pos_word_dict.txt" #wprd to pos prob dict
word_curpos_path = "/home/stufs1/nitigupta/628/pickled_ds/pos_curword_dict.txt" #cur wprd to cur pos prob dict
output_path = "/home/stufs1/nitigupta/628/pickled_ds/sents_with_predict_tags.txt" #cur wprd to cur pos prob dict

df = pd.DataFrame.from_csv(path,header=None,index_col=None,sep='\t')
test_data_df = pd.DataFrame.from_csv(test_data_path,header=None,index_col=None,sep='\t')


# <codecell>

L = len(df)
def make_index(l,d):
    for i in range (0,len(l)):
        d[l[i]]=i
        
def save_to_disk(ds,path):
    print("Pickling to... ",path)
    dumpfile=open(path,'wb')
    pickle.dump(ds,dumpfile)
    dumpfile.close()
    print("Done Pickling...")
    
def load_from_disk(path):
    print("Unpickling...")
    unpickled_file=open(path,'rb')
    ds=pickle.load(unpickled_file)
    print("Done Pickling...")
    return ds

# <codecell>

pos_tags_dict = dict()
for i in range (0,L):
    #print(df[][])
    pos_tags_dict[df[1][i]]=1
sorted(pos_tags_dict, key=pos_tags_dict.get)

pos_tags_l = pos_tags_dict.keys()
del pos_tags_dict
pos_tags_l.sort()

pos_tags_index_mapper = dict()
make_index(pos_tags_l,pos_tags_index_mapper)
del pos_tags_l

pos_tags_count = len(pos_tags_index_mapper)

# <codecell>

word_dict = dict()
for i in range (0,L):
    #print(df[][])
    word_dict[df[0][i]]=1
sorted(word_dict, key=word_dict.get)

word_l = word_dict.keys()
del word_dict
word_l.sort()

word_index_mapper = dict()
make_index(word_l,word_index_mapper)
del word_l
word_count = len(word_index_mapper)

# <codecell>

def set_prob(arr,count1,count2):
    
    column_sum = np.zeros(count2)
    for i in range(0,count1):
        for j in range(0,count2):
            column_sum[j]+=arr[i][j]
    
    for i in range(0,count1):
        for j in range(0,count2):
            arr[i][j] = arr[i][j]/column_sum[j]
            

# <codecell>

'''
bi_arr = np.zeros( ( pos_tags_count,pos_tags_count) )

for i in range(0,(L-1)):
    #print(i)
    bi_arr[pos_tags_index_mapper[df[1][i]]][pos_tags_index_mapper[df[1][i+1]]]+=1

set_prob(bi_arr,len(bi_arr),len(bi_arr))
total=0
for i in range(0,len(bi_arr)):
    total+=bi_arr[i][20]
print(total)

save_to_disk(bi_arr,bi_arr_path)
'''
#d = load_from_disk(bi_arr_path)

# <codecell>

#import seaborn as sns
bi_word_dict = dict()

def update_dict(d):
    
    if w1 not in d:
        d[w1]= {}
        d[w1][w2]=1
    else:
        if w2 not in d[w1]:
            d[w1][w2]=1
        else:
            d[w1][w2]+=1
def convert_to_prob(d):
    for outer_key in d:
        count = 0
        for inner_key in d[outer_key]:
            count+=d[outer_key][inner_key]
        for inner_key in d[outer_key]:
            d[outer_key][inner_key] = d[outer_key][inner_key]/count

# <codecell>
'''
bi_word_dict = dict()   

for i in range(0,(L-1)):
    
    w2=df[0][i]
    w1=df[0][i+1]
    if type(w1)==str:
        w1=w1.lower()
    if type(w2)==str:
        w2=w2.lower()
    update_dict(bi_word_dict)


#if probabilities are required, execute this also:
convert_to_prob(bi_word_dict)

save_to_disk(bi_word_dict,bi_word_path)
'''
# <codecell>

'''
pos_word_dict= dict()
for i in range(0,L):   
    w1=df[1][i] #pos
    w2=df[0][i]
    if type(w2)==str:
        w2=w2.lower()
    update_dict(pos_word_dict)

save_to_disk(pos_word_dict,pos_word_dict_path)
'''
# <codecell>
'''
pos_tags_prob = np.zeros(pos_tags_count)


for i in range(0,L):
    pos_tags_prob[pos_tags_index_mapper[df[1][i]]]+=1


pos_tags_prob=pos_tags_prob/L

save_to_disk(pos_tags_prob,pos_prob_path)
'''

# <codecell>

'''
word_arr = np.zeros( (word_count,pos_tags_count) )

for i in range(0,(L-1)):
    #print(i)
    word_arr[word_index_mapper[df[0][i]]][pos_tags_index_mapper[df[1][i+1]]]+=1 
    
set_prob(word_arr,word_count,pos_tags_count)

total=0
for i in range(0,pos_tags_count):
    total+=word_arr[20][i]
print(total)

save_to_disk(word_arr,word_pos_path)
'''
# <codecell>
'''
word_tag_arr = np.zeros( (word_count,pos_tags_count) )

for i in range(0,(L-1)):
    #print(i)
    word_tag_arr[word_index_mapper[df[0][i]]][pos_tags_index_mapper[df[1][i]]]+=1 
    
set_prob(word_tag_arr,word_count,pos_tags_count)

total=0
for i in range(0,pos_tags_count):
    total+=word_tag_arr[10][i]
print(total)

save_to_disk(word_tag_arr,word_curpos_path)
'''
# <codecell>
'''
tri_arr = np.zeros( (pos_tags_count,pos_tags_count,pos_tags_count) )

for i in range(0,(L-2)):
    #print(i)
    tri_arr[pos_tags_index_mapper[df[1][i]]][pos_tags_index_mapper[df[1][i+1]]][pos_tags_index_mapper[df[1][i+2]]]+=1

tri_column_sum=np.zeros(pos_tags_count)

#column_sum = np.zeros(len(tri_arr))
for k in range(0,len(tri_arr)):
    for i in range(0,len(tri_arr)):
        for j in range(0,len(tri_arr)):        
            tri_column_sum[k]+=tri_arr[i][j][k]

for k in range(0,len(tri_arr)):
    for i in range(0,len(tri_arr)):
        for j in range(0,len(tri_arr)):
            tri_arr[i][j][k] = tri_arr[i][j][k]/(tri_column_sum[k]+1)

total=0.0
for i in range(0,len(tri_arr)):
    for j in range(0,len(tri_arr)):
        total+=tri_arr[i][j][20]
    
print(total)
print(tri_arr[20][20][20])

save_to_disk(tri_arr,tri_arr_path)
'''
# <codecell>
'''
bi_arr = load_from_disk(bi_arr_path)
tri_arr = load_from_disk(tri_arr_path)
word_arr = load_from_disk(word_pos_path)

L_test_data = len(test_data_df)
counter = 1
output = []
sent = []
sents = []
for i in range(L_test_data):
        if (test_data_df[0][i] == "."):
            sents.append(copy.deepcopy(sent))
            #print sent
            del sent[:]
            continue
	word = test_data_df[0][i]
        tag = test_data_df[1][i]
        #print word,tag
        sent.append((word,tag))

        
for line in sents:
    if (len(line) <= 0):
	continue
    #print line
    local_min = 999999
    local_pos = 1
    # Get the probabs and identify the missing location
    for i in range(1,len(line)):
        if (i == 1):
            (prev_word, prev_tag) = line[i-1]
            (cur_word, cur_tag) = line[i]
            prob_prevtag_given_curtag = bi_arr[pos_tags_index_mapper[cur_tag],pos_tags_index_mapper[prev_tag]]*1000
            prob_curtag_given_prevtag = bi_arr[pos_tags_index_mapper[prev_tag],pos_tags_index_mapper[cur_tag]]*1000
            prob_curtag_given_prevtwotags = 0
            prob_curtag_given_prevword = 0.001
	    if (word_index_mapper.has_key(prev_word)):
		prob_curtag_given_prevword = word_arr[word_index_mapper[prev_word],pos_tags_index_mapper[cur_tag]]*1000
        else:
            (prev_prev_word, prev_prev_tag) = line[i-2]
            (prev_word, prev_tag) = line[i-1]
            (cur_word, cur_tag) = line[i]
            prob_prevtag_given_curtag = bi_arr[pos_tags_index_mapper[cur_tag],pos_tags_index_mapper[prev_tag]]*1000
            prob_curtag_given_prevtag = bi_arr[pos_tags_index_mapper[prev_tag],pos_tags_index_mapper[cur_tag]]*1000
            prob_curtag_given_prevtwotags = tri_arr[pos_tags_index_mapper[prev_prev_tag],pos_tags_index_mapper[prev_tag],pos_tags_index_mapper[cur_tag]]*1000
            prob_curtag_given_prevword = 0.001
	    if (word_index_mapper.has_key(prev_word)):
		prob_curtag_given_prevword = word_arr[word_index_mapper[prev_word],pos_tags_index_mapper[cur_tag]]*1000
        #print prob_curtag_given_prevtag, prob_curtag_given_prevtwotags, prob_curtag_given_prevword
        total_prob = prob_prevtag_given_curtag + prob_curtag_given_prevtag + prob_curtag_given_prevtwotags + 10*prob_curtag_given_prevword
        if (total_prob < local_min):
            local_min = total_prob
            local_pos = i
    
    # Get the missing tag now that the location is known
    
    #print new_line
    max_prob = 0
    max_prob_tag = ""
    for tag in pos_tags_index_mapper:
        print "local_pos : ",local_pos,"LENGTH ",len(line)
        (prev_word, prev_tag) = line[local_pos-1]
        if (word_index_mapper.has_key(prev_word) == False):
          continue
        prob_tag_given_prevtag = bi_arr[pos_tags_index_mapper[prev_tag],pos_tags_index_mapper[tag]]*1000
        prob_tag_given_prevword = word_arr[word_index_mapper[prev_word],pos_tags_index_mapper[tag]]*1000
        total_prob = prob_tag_given_prevtag + prob_tag_given_prevword
        if (total_prob > max_prob):
            max_prob = total_prob
            max_prob_tag = tag
    list1 = line[:local_pos]
    list1.append(('!missing!', max_prob_tag))
    list2 = line[local_pos:]
    new_line = list1 + list2
    output.append(new_line)
    #print new_line
    #print

save_to_disk(output, output_path)    
'''
#print(bi_arr[pos_tags_index_mapper['VB'],pos_tags_index_mapper['MD']]*1000)
#print(bi_arr[pos_tags_index_mapper['MD'],pos_tags_index_mapper['VB']]*1000)
#print(tri_arr[pos_tags_index_mapper['NN'],pos_tags_index_mapper['MD'],pos_tags_index_mapper['VB']]*1000)
#print(word_arr[word_index_mapper['may'],pos_tags_index_mapper['VB']]*1000)
# <codecell>

tagged_sents = load_from_disk(output_path)
#tagged_sents = output
pos_prob = load_from_disk(pos_prob_path)
tri_arr = load_from_disk(tri_arr_path)
word_arr = load_from_disk(word_pos_path)
pos_word_dict = load_from_disk(pos_word_dict_path)
bi_word = load_from_disk(bi_word_path)
word_curpos = load_from_disk(word_curpos_path)
output_sents = []
#print "asdsadddddsadasd",tagged_sents
#print pos_word_dict
#print bi_word['back']
#print word_curpos
counter = 0;

for line in tagged_sents:
    counter = counter + 1
    print "Sentence :",counter
    #print line
    local_max = 0
    missing_word = "!missing!"
    missing_tag = ""
    local_pos = 0
    new_line = []
    for i in range(1,len(line)):
        (cur_word, cur_tag) = line[i]
        if (cur_word == "!missing!"):
            if (cur_tag == '' or cur_tag == None or cur_tag == "nan"):
              cur_tag = "VB"
            if (pos_word_dict.has_key(cur_tag) == False):
               cur_tag = "VB"
            word_list = pos_word_dict[cur_tag]
            for word in word_list.keys():
                if (word_index_mapper.has_key(word) == False):
                    continue
                
                if (i == 1):
                    (prev_word, prev_tag) = line[i-1]
                    if (prev_tag == None or prev_tag == "nan"):
                      prev_tag = "VB"
                    prob_curtag = pos_prob[pos_tags_index_mapper[cur_tag]]*1000
                    prob_curtag_given_curword = word_curpos[word_index_mapper[word]][pos_tags_index_mapper[cur_tag]]*1000
                    prob_curtag_given_prevtwotags = 1
                    prob_curword_given_prevword = 0.001
                    if (bi_word.has_key(prev_word)):
                        dict_list = bi_word[prev_word]
                        if (dict_list.has_key(word)):
                            prob_curword_given_prevword = bi_word[prev_word][word]*1000                                               
                    final_prob = prob_curword_given_prevword*((prob_curtag_given_curword*prob_curtag_given_prevtwotags)/float(prob_curtag))
                else:
                    (prev_prev_word, prev_prev_tag) = line[i-2]
                    (prev_word, prev_tag) = line[i-1]
                    if (prev_tag == None or prev_tag == "nan"):
                      prev_tag = "VB"
                    if (prev_prev_tag == None or prev_prev_tag == "nan"):
                      prev_prev_tag = "VB"
                    #print prev_prev_tag, prev_tag, cur_tag
                    prob_curtag = pos_prob[pos_tags_index_mapper[cur_tag]]*1000
                    prob_curtag_given_curword = word_curpos[word_index_mapper[word]][pos_tags_index_mapper[cur_tag]]*1000
                    if (pos_tags_index_mapper.has_key(prev_prev_tag) == False):
			prev_prev_tag = "NN"
                    if (pos_tags_index_mapper.has_key(prev_tag) == False):
			prev_tag = "NN"
                    if (pos_tags_index_mapper.has_key(cur_tag) == False):
			cur_tag = "NN"
                    prob_curtag_given_prevtwotags = tri_arr[pos_tags_index_mapper[prev_prev_tag],pos_tags_index_mapper[prev_tag],pos_tags_index_mapper[cur_tag]]*1000
                    prob_curword_given_prevword = 0.001
                    if (bi_word.has_key(prev_word)):
                        dict_list = bi_word[prev_word]
                        if (dict_list.has_key(word)):
                            prob_curword_given_prevword = bi_word[prev_word][word]*1000 
                    final_prob = prob_curword_given_prevword*((prob_curtag_given_curword*prob_curtag_given_prevtwotags)/float(prob_curtag))
                if (final_prob > local_max):
                    local_max = final_prob
                    missing_word = word
                    local_pos = i
                    missing_tag = cur_tag
            break
    list1 = line[:local_pos]
    if (missing_word == "!missing!"):
      missing_word = " "
    list1.append((missing_word, missing_tag))
    list2 = line[(local_pos+1):]
    #list2.append((local_max, missing_word))
    new_line = list1 + list2
    output_sents.append(new_line)

out_path = "/home/stufs1/nitigupta/628/data/final_submission.txt"    
## try to open the file ##
try:
  f = open(out_path, 'w')
except:
  print "Cant open file"
  sys.exit(0)

index = 1;

print f.write("\"id\",\"sentence\"")
for line in output_sents:
   print "are we here ",line
   final_string = ""
   for tup in line:
	(word, tag) = tup
	final_string = final_string + str(word) + " "
   final_string = str(index) + ",\"" + final_string + "\"\n"
   index = index+1
   print f.write(final_string)

f.close()
# <codecell>


