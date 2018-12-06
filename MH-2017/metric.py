
# coding: utf-8

# In[56]:


import pandas as pd
import string
import word2vec_m as wv
# import weather_api
from gensim.models import Word2Vec
import numpy as np
import sys
import data_cleaner

import similarity

dimen = int(sys.argv[1])
k = int(sys.argv[2])
a = float(sys.argv[3])
pca_text = str(sys.argv[4])

file_name = 'metric_test_dimen{}_k{}_a{}.csv'.format(dimen, k, a)

u, v, t, new_maharashtra, maharashtra = wv.pre('asf-all-train.csv')#('all_files.csv')

# print t

# train_size = (len(new_maharashtra)*8)/10
# test_size = len(new_maharashtra) - train_size

train_data = new_maharashtra
test_data = pd.read_csv('asf-all-test.csv')['QueryText'] #maharashtra['Query'][train_size:]

wv.word2vec_QAmodel(u, v, t, train_data, dimen, a, pca=pca_text)

district = 'Pune'
state = 'Maharashtra'
 
#### Answer #####

word2vec_value = np.load('word2vec_value.npy')
model = Word2Vec.load('model_word2vec.bin')

count_lesk = 0
count_jaccard = 0
count_lesk_threshold = 0
count_jaccard_threshold = 0
# threshold_lesk = 0.95
# threshold_jaccard = 0.8
threshold_lesk = 0.8
threshold_jaccard = 0.7
total_count = 0

vayu = []
c = 0

# test_data = pd.read_csv('metric_test.csv')

for i,query in enumerate(list(test_data)):

    c+=1
    print c

    query = query.lower()
    input_list = [district,state,query]

    ind = wv.test_metric(u,v,t,query,dimen,k,a,model,word2vec_value,pca=pca_text)
    pdf = maharashtra.reset_index()

    query_list = data_cleaner.sentence_cleaner(query)
    fin_index = wv.entity(ind, query_list, pdf)

    lesk_score = similarity.compute_lesk_score(query, pdf['Query'][ind[fin_index]])
    jaccard_score = similarity.compute_jaccard_sim(query, pdf['Query'][ind[fin_index]])

    # print pdf['Query'][ind[fin_index]]

    if lesk_score > threshold_lesk:
          count_lesk_threshold += 1

    if lesk_score > 0:
          count_lesk += 1

    if jaccard_score > threshold_jaccard:
          count_jaccard_threshold += 1

    if jaccard_score > 0:
          count_jaccard += 1

print "total count:", total_count
print "count_jaccard_threshold: ", count_jaccard_threshold
print "count_lesk_threshold: ", count_lesk_threshold



#top-1
#top-3
#top-5



#     c += 1
# #     print c
    
#     for j in ind:
#        total_count += 1
#        lesk_score = similarity.compute_lesk_score(query, pdf['Query'][j])
#        jaccard_score = similarity.compute_jaccard_sim(query, pdf['Query'][j])

#        if lesk_score>threshold_lesk:
#               count_lesk_threshold += 1
#        if lesk_score>0:
#               count_lesk += 1
#        if jaccard_score>threshold_jaccard:
#               count_jaccard_threshold += 1
#        if jaccard_score>0:
#               count_jaccard += 1

#        vayu.append([lesk_score, jaccard_score, query, list(maharashtra['Ans'][train_size:])[i], pdf['Query'][j], pdf['Ans'][j]])

# print "total count:", total_count
# print "count_jaccard_threshold: ", count_jaccard_threshold
# print "count_lesk_threshold: ", count_lesk_threshold


# vayu = pd.DataFrame(vayu)
# vayu.to_csv('./metric/{}'.format(file_name))






# wv.print_ans(ind, maharashtra, k)