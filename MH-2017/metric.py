
# coding: utf-8

# In[56]:


import pandas as pd
import string
import word2vec_m as wv
import weather_api
from gensim.models import Word2Vec
import numpy as np
import sys

dimen = int(sys.argv[1])
k = int(sys.argv[2])
a = float(sys.argv[3])
pca_text = str(sys.argv[4])

u, v, t, new_maharashtra, maharashtra = wv.pre('all_files.csv')

print t

train_size = (len(new_maharashtra)*8)/10
test_size = len(new_maharashtra) - train_size

train_data = new_maharashtra[:train_size]
test_data = maharashtra['Query'][train_size:]

wv.word2vec_QAmodel(u, v, t, train_data, dimen, a, pca=pca_text)

district = 'Pune'
state = 'Maha'
 
#### Answer #####

word2vec_value = np.load('word2vec_value.npy')
model = Word2Vec.load('model_word2vec.bin')

vayu = []
c = 0
for i,query in enumerate(list(test_data)[:100]):

    query = query.lower()
    input_list = [district,state,query]

    ind = wv.test_metric(u,v,t,query,dimen,k,a,model,word2vec_value,pca=pca_text)

    pdf = maharashtra.reset_index()
    c += 1
    print c
    
    for j in ind:
       vayu.append([query, list(maharashtra['Ans'][train_size:])[i], pdf['Query'][j], pdf['Ans'][j]])

vayu = pd.DataFrame(vayu)
vayu.to_csv('metric_test_1.csv')





# wv.print_ans(ind, maharashtra, k)

# In[55]:




