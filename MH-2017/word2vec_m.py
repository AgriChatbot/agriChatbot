
# coding: utf-8

# In[86]:


import pandas as pd 
from gensim.models import Word2Vec
from nltk.corpus import stopwords 
import numpy as np
from nltk.probability import FreqDist, MLEProbDist
import scipy
from scipy.linalg import svd
from sklearn.decomposition import PCA
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import similarity
import data_cleaner
import warnings
warnings.filterwarnings("ignore")
# In[92]:

def pre(filename):
    stop_words = set(stopwords.words('english'))
    maharashtra = pd.read_csv(filename,index_col = False).sort_values('questions')

    query = list(maharashtra['questions'])
    kccans = list(maharashtra['answers'])
    district = list(maharashtra['district'])
    state = list(maharashtra['state'])

    final = []
    for i,w in enumerate(query):
        if query[i] != kccans[i]:
            if type(query[i]) == str:
                query[i] = query[i].replace('?','')
                query[i] = query[i].replace('(','')
                query[i] = query[i].replace(')','')

        if type(query[i]) != float:
            if type(kccans[i][0]) != float:
                row = [query[i].lower().strip(),kccans[i].lower().strip(),district[i].lower().strip(),state[i].lower().strip()]
                if row[0].isalpha() or len(row[0])>5:
                    final.append(row)

    maharashtra = pd.DataFrame(final)
    maharashtra.columns = ['Query','Ans','District','State']
    # maharashtra = maharashtra.sample(frac=1)

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

        #new_maharashtra.append([maharashtra['District'][w],maharashtra["State"][w]] + temp)
        new_maharashtra.append(temp)
 
    t = FreqDist(all_words)
    u = FreqDist(district)
    v = FreqDist(state)
    
    return u,v,t,new_maharashtra,maharashtra


# In[88]:


def word2vec_QAmodel(u,v,t,new_maharashtra,dimen,a,**pca):
        
    for key, value in pca.items(): 
        key = value

    model = Word2Vec(new_maharashtra, min_count = 1,size=dimen)

    word2vec_value = []
    for i in new_maharashtra:
        value = np.array([0.0 for k in range(dimen)])
        count = len(i)
        c = 0
        for j in i:
            c += 1
#             if c in [1]:
#                 factor = 1/(1 + u[j]/u.N())
#             if c in [2]:
#                 factor = 0.0001/(0.0001 + v[j]/v.N())
#             if c in [3,4,5]:
#                 factor = 1#a/(a + t[j]/t.N())
            factor = a/(a + t[j]/t.N()) #1
            # factor = 1000
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

    model.save('model_word2vec.bin')
    word2vec_value = np.matrix(word2vec_value)
    
    np.save('word2vec_value',word2vec_value)


# In[89]:


def test_query(u,v,t,input_list,dimen,no_similar,a,**pca):
    stop_words = set(stopwords.words('english')) 
    
    word2vec_value = np.load('word2vec_value.npy')
    model = Word2Vec.load('model_word2vec.bin')
    
    district = input_list[0]
    state = input_list[1]
    sent = input_list[2]

    sent = data_cleaner.sentence_cleaner(sent)
    # sent_words = [district,state] + sent.split(" ")
    sent_words = sent.split(" ")

    sent_value = np.array([0.0 for k in range(dimen)])

    sent_new = []
    count = 0

    for i in sent_words:
        if i not in stop_words:
            try:
                # if count in [0]:
                #     factor = a/(a + u[i]/u.N())
                # if count in [1]:
                #     factor = a/(a + v[i]/v.N())
                # if count in [2,3,4]:
                #     factor = a/(a + t[i]/t.N())
                factor = a/(a + t[i]/t.N())#1
                # factor = 1000
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
    
    return ind

def test_metric(u,v,t,sent,dimen,no_similar,a,model,word2vec_value,**pca):
    stop_words = set(stopwords.words('english')) 
    
    # word2vec_value = np.load('word2vec_value.npy')
    # model = Word2Vec.load('model_word2vec.bin')
    
    # district = input_list[0]
    # state = input_list[1]
    # sent = input_list[2]

    #sent_words = [district,state] + sent.split(" ")
    sent_words = sent.split(" ")

    sent_value = np.array([0.0 for k in range(dimen)])

    sent_new = []
    count = 0

    for i in sent_words:
        if i not in stop_words:
            try:
                # if count in [0]:
                #     factor = a/(a + u[i]/u.N())
                # if count in [1]:
                #     factor = a/(a + v[i]/v.N())
                # if count in [2,3,4]:
                #     factor = a/(a + t[i]/t.N())
                factor = a/(a + t[i]/t.N()) #1
                # factor = 1000
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
    
    return ind


# In[90]:


def entity(ind, input_list, pdf):

    pattern = 'NP: {<DT>?<JJ>*<NN>}' #'NP: {<NNS>?<NN>}'
    cp = nltk.RegexpParser(pattern)


    to_input = []
    sent = input_list
    sent = nltk.pos_tag(sent)
    cs = cp.parse(sent)

    for i in cs:
        try:
            for j in range(10):
                if i[j][1] == "NN" or i[j][1] == "JJ":
                    to_input.append(i[j][0])

        except:
            pass  
            
    dictionary = ['farmer','krishi','dose', 'spray', 'information', 'market', 'rate', 'fertiliser', 'growth', 'variety', 'management']
    
    for i in dictionary:
        if i in to_input:
            to_input.remove(i)
    
    choose = []
    
    for i in ind:
        to = []
        sent = pdf['Query'][i]
        sent = nltk.word_tokenize(sent)
        sent = nltk.pos_tag(sent)
        cs = cp.parse(sent)

        for k in cs:
            try:
                for j in range(10):
                    if k[j][1] == "NN" or k[j][1] == "JJ":
                        to.append(k[j][0])

            except:
                pass

        for j in dictionary:
            if j in to:
                to.remove(j)

        # to = ' '.join(to)
        # to_input = ' '.join(to_input)

        to = set(to)
        to_input = set(to_input)
        # lesk = similarity.compute_lesk_score(to_input, to)

        sim = to_input.intersection(to)
        choose.append(len(sim))
        # if lesk > 0:
        #     flg = i
        #     break

    hero = -1
    flg = -1
    
    for i,w in enumerate(choose):
        if w > hero:
            hero = w
            flg = i
            print hero, flg

    return flg
            
        
def find_best_answer(question,ans_list):
    max_ls = 0
    correct_answer = ans_list[0]

    for ans in ans_list:
        lesk_score = similarity.compute_lesk_score(question, ans)

        if lesk_score > max_ls:
            max_ls = lesk_score
            correct_answer = ans

    return correct_answer
# In[91]:

def print_ans(ind, pdf, k):
    
    #pdf = maharashtra
    pdf = pdf.reset_index()
    
    # print 'Top-%d\n\n'%(k)
    exec("ans_list = %s"%(pdf['Ans'][ind[0]]))

    query  = pdf['Query'][ind[0]]

    ans = find_best_answer(query, ans_list)
    # for i in ind:
    # print 'Question: %s\nAnswer: %s\n\n'%(pdf['Query'][i],pdf['Ans'][i])
    print 'Question: %s\nAnswer: %s\n\n'%(pdf['Query'][ind[0]],ans)

# u,v,t,new_maharashtra,maharashtra = preproc('all_files.csv')

# word2vec_QAmodel(u,v,t,new_maharashtra,50,0.001,pca='Yes')

# ind = test_query(u,v,t,['pune','maharashtra','blight attack on paddy'],50,5,0.001,pca='Yes')

# print_ans(ind, maharashtra, 5)

