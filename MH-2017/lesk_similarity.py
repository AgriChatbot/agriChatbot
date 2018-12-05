import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize


def pre_proccess(sentence):
    processed_sent = []
    # tokenize words of the sentence
    words = word_tokenize(sentence)
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
    context_bag1 = set(lesk_context_bag(sentence1))
    context_bag2 = lesk_context_bag(sentence2)
    lesk_score = len(context_bag1.intersection(context_bag2))
    
    # lesk_score = 0
    # for w in context_bag2:
    #     if w in sentence1:
    #         lesk_score +=1
    
    return lesk_score



c1 = compute_lesk_score("The interest rate for bonds is low", "banker drive car on the bank of the road")
c2 = compute_lesk_score("The interest rate for bonds is low", "banker do not deposit money in bank")

print c1, c2