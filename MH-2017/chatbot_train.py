
# coding: utf-8

# In[2]:


import pandas as pd
import string
import word2vec_m as wv
import sys


# In[3]:

dimen = int(sys.argv[1])
a = float(sys.argv[2])
pca_text = str(sys.argv[3])

u, v, t, new_maharashtra, maharashtra = wv.pre('all_files.csv')
wv.word2vec_QAmodel(u,v,t,new_maharashtra,dimen,a,pca=pca_text)

