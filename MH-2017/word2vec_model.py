
# coding: utf-8

# In[135]:


import pandas as pd 
from gensim import models
from nltk.corpus import stopwords 
import numpy as np
from nltk.probability import FreqDist, MLEProbDist
import scipy
from scipy.linalg import svd
from sklearn.decomposition import PCA

# stop_words = set(stopwords.words('english')) 

# maharashtra = pd.DataFrame()

# for i in range(1,36):
#     tempo = pd.read_csv('11_%d.csv'%(i))
#     maharashtra = pd.concat([tempo,maharashtra])


# In[185]:


def word2vec_QAmodel(input_list,data,dimen,no_similar,a,**pca):
    
    stop_words = set(stopwords.words('english')) 
    maharashtra = pd.DataFrame()

    for i in range(1,36):
        tempo = pd.read_csv('11_%d.csv'%(i))
        maharashtra = pd.concat([tempo,maharashtra])

        
        
        
        
        
        
    for key, value in pca.items(): 
        key = value
    
    query = list(data['QueryText'])
    kccans = list(data['KCCAns'])
    district = list(data['DistrictName'])
    state = list(data['StateName'])

    final = []
    for i,w in enumerate(query):
        if query[i] != kccans[i]:
            if type(query[i]) == str:
                query[i] = query[i].replace('?','')
                query[i] = query[i].replace('(','')
                query[i] = query[i].replace(')','')

        if type(query[i]) != float:
            if type(kccans[i]) != float:
                row = [query[i].lower().strip(),kccans[i].lower().strip(),district[i].lower().strip(),state[i].lower().strip()]
                if row[0].isalpha() or len(row[0])>5:
                    final.append(row)

    maharashtra = pd.DataFrame(final)
    maharashtra.columns = ['Query','Ans','District','State']

    main = []
    for w in list(maharashtra['Query']):
        main.append(w.split(' '))

    new_maharashtra = []
    all_words = []
    for w,i in enumerate(main):
        temp = []
        for j in i:
            if j not in stop_words:
                temp.append(j)
                all_words.append(j)

        new_maharashtra.append([maharashtra['District'][w],maharashtra["State"][w]] + temp)

    model = models.Word2Vec(new_maharashtra, min_count = 1,size=dimen)

    t = FreqDist(all_words)
    u = FreqDist(district)
    v = FreqDist(state)

    word2vec_value = []
    for i in new_maharashtra:
        value = np.array([0.0 for k in range(dimen)])
        count = len(i)
        c = 0
        for j in i:
            c += 1
            if c in [1]:
                factor = 1/(1 + u[j]/u.N())
            if c in [2]:
                factor = 0.0001/(0.001 + v[j]/v.N())
            if c in [3,4,5]:
                factor = a/(a + t[j]/t.N())
            value += model[j]*factor

        value = value/count
        word2vec_value.append(value)
    
    if pca == 'Yes':
        X = np.array(word2vec_value)
        pca = PCA()
        pca.fit(X)

        u = pca.components_[0]
        u_t = np.transpose(u)
        u_ut = np.matmul(u,u_t)

        new_word2vec_value = []
        for i,w in enumerate(word2vec_value):
            sub = w - u_ut*np.array(w)
            new_word2vec_value.append(sub)

        word2vec_value = new_word2vec_value

           
    district = input_list[0]
    state = input_list[1]
    sent = input_list[2]

    sent_words = [district,state] + sent.split(" ")

    sent_value = np.array([0.0 for k in range(dimen)])

    sent_new = []
    count = 0

    for i in sent_words:
        if i not in stop_words:
            try:
                if count in [0]:
                    factor = 1/(1 + u[i]/u.N())
                if count in [1]:
                    factor = 0.0001/(0.001 + v[i]/v.N())
                if count in [2,3,4]:
                    factor = a/(a + t[i]/t.N())
                sent_value += model[i]*factor
                count += 1

            except:
                count += 1
                continue

    sent_value = sent_value/count
    
    if pca == 'Yes':
        sent_value = sent_value - u_ut*np.array(sent_value)

    all_dist = []
    for i in word2vec_value:
        dist = scipy.spatial.distance.cosine(i,sent_value)
        all_dist.append(dist)

    k = no_similar
    ind = np.argpartition(all_dist, k)[:k]
    
    pdf = maharashtra
    pdf = pdf.reset_index()
    
    print 'Top-%d\n\n'%(k)
    for i in ind:
        print 'Question: %s\n Answer: %s\n\n'%(pdf['Query'][i],pdf['Ans'][i])
    
   # return ind


# In[186]:


#word2vec_QAmodel(['pune','maharashtra','attack of wilt'],maharashtra,50,3,0.001,pca='Yes')

