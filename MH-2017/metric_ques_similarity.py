import pandas as pd
import string
import word2vec_m as wv
import sys
import data_cleaner
import jaccard_similarity

dimen = 50 # int(sys.argv[1])
k = 1 # int(sys.argv[2])
a = 0.1 # float(sys.argv[3])
pca_text = 'yes'#str(sys.argv[4])



#Chatbot
print '\nHi!, I am AgriBot'
print 'I am here to help you out with agriculture queries.\n'
print 'May I know your name?'

name = 's'#raw_input('')
name = str(name)

name = name.split(' ')[0]

print '\nSo %s, How can I help you?'%(name)

query = 'what is the market rate of onions?' #raw_input('')

print '\nSure, which district are you from?'
district = 'pune' #raw_input('')

print '\nAnd which state?'
state = 'maharashtra' #raw_input('')

print '\nGive me a second\n\n'


#### test #####

predicted_ans_list = []
predicted_query_list = []

if 'weather' in query:
    print 'weather query not to be tested'
else:
    query = query.lower()
    query_t = data_cleaner.sentence_cleaner(query)
    input_list = [district,state,query_t]
    print input_list

    u, v, t, new_maharashtra, maharashtra = wv.pre('all_files.csv')
    ind = wv.test_query(u, v, t, input_list,dimen,k,a,pca=pca_text)

    # wv.print_ans(ind, maharashtra, k)
    
    maharashtra = maharashtra.reset_index()
    for i in ind:
        predicted_query_list.append(maharashtra['Query'][i])
        # print 'Answer: %s\n\n'%(maharashtra['Ans'][i])
        exec('mah_list=%s' % (maharashtra['Ans'][i]))
        for j in mah_list:
            predicted_ans_list.append(j)

que_jaccard_score_list = []

for p_que in predicted_query_list:
    pred_q_clean = data_cleaner.sentence_cleaner(p_que.lower())

    list_query = query_t.split()
    list_pred = pred_q_clean.split()
    print '\n\noriginal:',list_query,'\n predicted', list_pred
    que_jaccard_score_list.append(jaccard_similarity.jaccard_sim(list_query, list_pred))


print que_jaccard_score_list, sum(que_jaccard_score_list)


test_ans = 'this is where the answer for given query from test data set comes'
test_ans = data_cleaner.sentence_cleaner(test_ans)
test_ans_list = test_ans.split()

ques_jaccard_scores = []
ans_jaccard_score_list = []
for predicted_ans in predicted_ans_list:
    predicted_ans = data_cleaner.sentence_cleaner(predicted_ans)
    predicted_ans_list = predicted_ans.split()
    
    ans_jaccard_score_list.append(jaccard_similarity.jaccard_sim(test_ans_list, predicted_ans_list))