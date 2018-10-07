# agriChatbot

The following project is an attempt to automate general farmer queries in India. The entire code has been written in Python. 

### Features
* A command line application, where general agriculture related queries can be asked.
* Has been trained using data multiple states, which means the answers may change, depending om the place.
* weathers queries has been linked to a weather API

### Improvements

* A complete UI for the following app
* analytics
* integration with other languages
* Developiong new train models for our app


## Data Collection

For data collection, we used the Indian goverment website, which has data for farmer telephone queries of multiple states. In order to scrape this, we used `puppeteer` , a JS library to automate it

```javascript
const puppeteer = require('puppeteer');
```

## Preprocessing

Our preprocessing, includes removing stop word, spell correct which replaces incorrect words and words which may not be English and lastlty stemming

```python
text[i] = spell_correct_functions.correction(text[i])


ps = PorterStemmer()
wn = WordNetLemmatizer()
```
## Training Model

Our training model uses an implementation of the `word2vec`.

```python
word2vec_value = []
for i in new_maharashtra:
    value = np.array([0.0 for k in range(50)])
    count = len(i)
    c = 0
    for j in i:
        c += 1
        if c in [1]:
            factor = 0.001/(0.001 + u[j]/u.N())
        if c in [2]:
            factor = 0.001/(0.001 + v[j]/v.N())
        if c in [3,4,5]:
            factor = 0.001/(0.001 + t[j]/t.N())
        value += model[j]*factor
    
    value = value/count
    word2vec_value.append(value)
```

## Result

On the way!


## Contributors

This project has been developed by:
- Naman Jain
- Pranjali Jain
- Jayakrishna Sahit
- Pratik Kayal


