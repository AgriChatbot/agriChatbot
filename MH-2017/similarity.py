import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize


def jaccard_sim(list_test, list_train):
    test = set(list_test)
    train = set(list_train)
    c = test.intersection(train)
    # return float(len(c)) / (len(test) + len(train) - len(c)) 
    return float(len(c)) / (len(test) + 1)


def pre_proccess(sentence):
    processed_sent = []
    # tokenize words of the sentence
    words = word_tokenize(sentence.lower())
    # Get stop words
    stop_words = set(stopwords.words("english"))

    # Remove stopwards and add lemmatized root form of words
    for w in words:
        if w not in stop_words:
                processed_sent.append(WordNetLemmatizer().lemmatize(PorterStemmer().stem(w)))

    return processed_sent


def lesk_context_bag(context_sentence):
    context_bag_list = []
    for w in context_sentence:
        for syn in wn.synsets(w):
            gloss = pre_proccess(str(syn.definition()))
            for w_g in gloss:
                # if w_g not in context_bag_list:
                context_bag_list.append(w_g)
    return context_bag_list


def compute_lesk_score(sentence1, sentence2):
    sentence1 = pre_proccess(sentence1)
    sentence2 = pre_proccess(sentence2)
    lesk_scores = {}
    context_bag1 = lesk_context_bag(sentence1)
    context_bag2 = set(lesk_context_bag(sentence2))

    lesk_score = 1.0*len(set(context_bag1).intersection(set(context_bag2)))/( 1 + len(set(context_bag1)))
    
    # lesk_score = 0
    # for w in context_bag2:
    #     if w in sentence1:
    #         lesk_score +=1
    
    return lesk_score

sent1 = "what is market rate of onion"
sent2 = "pesticide rate of onion is 5"

c = compute_lesk_score(sent1, sent2)
j = jaccard_sim(sent1.lower().split(), sent2.lower().split())

print c,j


# Test

import pandas as pd
def test():
    df = pd.read_csv('metric_test.csv')
    test_question = list(df['0'] )
    predicted_question = list(df['2'] )

    list_jaccard = []
    list_lesk = []

    count_jaccard = 0
    count_jaccard_half = 0
    count_lesk = 0
    count_lesk_half = 0


    for test,pred in zip(test_question, predicted_question):
        js = jaccard_sim(test.lower().split() , test.lower().split())
        lesk_s = compute_lesk_score(test, pred)
        list_jaccard.append(js)
        list_lesk.append(lesk_s)
        
        if lesk_s>0.95:
            count_lesk_half += 1
        if lesk_s>0:
            count_lesk += 1
        if js>0.8:
            count_jaccard_half += 1
        if js>0:
            count_jaccard += 1

    df_result = pd.read_csv('metric_result.csv')
    ground_truth = list(df_result['Truth'])

    # print list_lesk, 
    # print list_jaccard
    # print ground_truth

    count_ground_truth = 0
    for i in ground_truth:
        count_ground_truth += i

    print count_ground_truth
    print count_lesk, count_jaccard
    print count_lesk_half, count_jaccard_half