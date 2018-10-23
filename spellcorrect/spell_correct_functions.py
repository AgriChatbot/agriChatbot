import re
from collections import Counter
import pandas as pd

import sqlite3

DATABASE='edit_distance.db'
conn =  sqlite3.connect(DATABASE)
conn.execute('CREATE TABLE IF NOT EXISTS edit_table_zero(INPUT_WORD TEXT , EDITED_WORD TEXT, UNIQUE(INPUT_WORD, EDITED_WORD))')
conn.execute('CREATE TABLE IF NOT EXISTS edit_table_one(INPUT_WORD TEXT , EDITED_WORD TEXT, UNIQUE(INPUT_WORD, EDITED_WORD))')
conn.execute('CREATE TABLE IF NOT EXISTS edit_table_two(INPUT_WORD TEXT , EDITED_WORD TEXT, UNIQUE(INPUT_WORD, EDITED_WORD))')
cur = conn.cursor()

dict_number = {0:'zero', 1:'one', 2:'two'}

def make_list_word(in_word, edit_distance_length):
    # print('SELECT EDITED_WORD FROM edit_table_{} WHERE INPUT_WORD = {}'.format(str(edit_distance_length), in_word))
    all_rows = cur.execute("SELECT EDITED_WORD FROM edit_table_{}".format(dict_number[edit_distance_length])+ " WHERE INPUT_WORD = (?)",(in_word,))

    list_edit_word = []
    for row in all_rows:
        list_edit_word.append(row[0])

    return list_edit_word


def add_words_db(in_word, list_edit_word, edit_distance_length):
    for word in list_edit_word:
    #     print("INSERT OR IGNORE INTO edit_table_{}".format(dict_number[edit_distance_length]+"(INPUT_WORD, EDITED_WORD) VALUES(?,?)"
        conn.execute("INSERT OR IGNORE INTO edit_table_{}".format(dict_number[edit_distance_length])+"(INPUT_WORD, EDITED_WORD) VALUES(?,?)",(in_word, word))
        conn.commit()


#  Create a list of agricultural words
crop_names = pd.read_csv('Cropnames_Indianlanguages.csv')
crop_common_name = []
for i in list(crop_names['English']):
    for j in str(i).lower().split():
        crop_common_name.append(j)

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('agri_corpus.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    # Check if word is a known agricultural term
    if word in crop_common_name:
        return(word)
    else:
        "Most probable spelling correction for word."
        return max(candidates(word), key=P)

def candidates(word):
    try:
        # print('Hi from DB')
        edit = {0: [], 1: [], 2: []}
        for i in edit:
            edit[i] = set( make_list_word(word, i) )
            # edit_0 = set( make_list_word(word, 0) )
            # edit_1 = set( make_list_word(word, 1) )
            # edit_2 = set( make_list_word(word, 2) )

    except:
        # print('DB not used')
        edit = {0: [], 1: [], 2: []}
        edit[0] = known([word])
        edit[1] = known(edits1(word)) 
        edit[2] = known(edits2(word))
        for i in edit:
            add_words_db(word, list(edit[i]), i)

    return (edit[0] or edit[1] or edit[2] or [word])




    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


# print(correction('menth'))
# print(correction('informatioon'))
# print(correction('informettion'))
# print(correction('wether'))
# print(correction('weatther'))
# print(correction('wither'))