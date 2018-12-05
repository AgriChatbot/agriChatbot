from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords as stopwords
import pandas as pd
import csv, operator
from spellcorrect import spell_correct_functions
import re

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
wn = WordNetLemmatizer()

# crop names in other languages
crop_names = pd.read_csv('../spellcorrect/Cropnames_Indianlanguages.csv')
crop_common_name = crop_names['English']
crop_hindi_name = crop_names['Hindi']
crop_hindi_eng_dict = {}
for eng, hin in zip(crop_common_name,crop_hindi_name):
    crop_hindi_eng_dict.update({str(hin).lower(): str(eng).lower()})


def sentence_cleaner(sent):
	text = re.findall(r'\w+' ,sent.lower())
	# print(text)
	p_text = []
	p_sent = ''
	for i,w in enumerate(text):
		if w in crop_hindi_eng_dict:
			text[i] = crop_hindi_eng_dict[w]
		else:
			text[i] = spell_correct_functions.correction(w)

		if w not in stop_words:
			p_text.append(w)
			p_sent += w + ' '
	return p_sent


# test for functions
# temp = sentence_cleaner('Hello darkness my old friend its nice to see you again')
# print(temp)