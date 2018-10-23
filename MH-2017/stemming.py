from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords as stopwords
import pandas as pd
import csv, operator
import spell_correct_functions
import re

Ishtopwords = set(stopwords.words('english'))


def data_cleaning():

    crop_names = pd.read_csv('Cropnames_Indianlanguages.csv')
    crop_common_name = crop_names['English']
    crop_hindi_name = crop_names['Hindi']
    crop_hindi_eng_dict = {}
    for eng, hin in zip(crop_common_name,crop_hindi_name):
        crop_hindi_eng_dict.update({str(hin).lower():str(eng).lower()})
    #crop_hindi_eng_dict

    data =[]
    for i in xrange(19,20):
        filename = 'data/'+str(i+1)+'.csv'
        with open(filename, 'r') as File: 
            reader = csv.DictReader(File)
            for row in reader:
                data.append(row)
                
    stopwords = ['about', '?', 'of', 'ask', 'in', 'of', 'at']
    ps = PorterStemmer()
    wn = WordNetLemmatizer()
    df = pd.DataFrame(data)[:100]
    # print df
    # print df


    # top_crops = ['cotton', 'sugarcane', 'onion', 'white gram', 'pomegranate', 'weed', 'groundnut', 'maize', 'tomato', 'brinjal', 
    #                 'soybean', 'sugarcane', 'wheat', 'turmeric', 'chilli', 'bengal mango', 'banana', 'pea', 'watermelon', 'cucumber',
    #                 'papaya', 'bhindi', 'lemon', 'sorghum', 'ginger', 'mosambi', 'coriander', 'custard', 'drumstick','cauliflower', 'coconut',
    #                 'grapes', 'pea', 'cabbage', 'guava', 'grass', 'mava', 'garlic', 'orange', 'caps', 'mango']


    # crops = df['Crop']
    # crop_count = {}
    # for i in xrange(len(crops)):
    #     if crops[i].lower() != 'other':
    #         top_crops.append(top_crops)
    #     try:
    #         crop_count[crops[i].lower()] += 1
    #     except:
    #         crop_count[crops[i].lower()] = 1

    # top_crops_from_cat =  sorted( crop_count.items(), key=operator.itemgetter(1), reverse=True)
    # crops = crops + top_crops
    # crop_list = list(set(crops))

    vocabulary = {}
    questions = df['QueryText']
    for qi in xrange(len(questions)):
        text = re.findall(r'\w+' ,questions[qi].lower())
        for i in xrange(len(text)):

            if text[i] in crop_hindi_eng_dict.keys():
                text[i] = crop_hindi_eng_dict[text[i]]
            else:
                text[i] = spell_correct_functions.correction(text[i])

            #text[i] = ps.stem(text[i].decode('utf-8'))
            #text[i] = wn.lemmatize(text[i].decode('utf-8'))
            if text[i] not in Ishtopwords:
                try:
                    vocabulary[text[i]] += 1
                except:
                    vocabulary[text[i]] = 1
        if len(text) > 0:
            questions[qi] = ' '.join(text)


    # voc_stats =  sorted(vocabulary.items(), key=operator.itemgetter(1), reverse=True)
    # print len(voc_stats)
    # print voc_stats

    df1 = df[['QueryText', 'KCCAns']]
    df1.drop_duplicates(inplace=True)
    df2 =  df1.sort_values(by=['QueryText'])

    df2.to_csv('edited_df_for_MH_upto_19.csv')

    for i in xrange(len(df2)):
        try:
            #print df2.loc[i, 'QueryText']
            if 'weather' in df2.loc[i, 'QueryText'].lower():
                df2.drop(i, inplace=True)
        except:
            None

    #print df2

    q_a = {}
    c = 0
    for r in range(len(df2)):
        if '??' not in df2.iloc[r]['QueryText']:
            try:
                if df2.iloc[r]['KCCAns'] not in q_a[df2.iloc[r]['QueryText']]:
                    q_a[df2.iloc[r]['QueryText']].append(df2.iloc[r]['KCCAns'])
                    c += 1
            except:
                q_a[df2.iloc[r]['QueryText']] = [df2.iloc[r]['KCCAns']]

    return q_a

