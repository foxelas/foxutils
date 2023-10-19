# Import / Install libraries

import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import RegexpTokenizer

from .utils import core_utils

disable_lemmatizer = bool(eval(core_utils.settings['RUN']['disable_lemmatizer']))
if not disable_lemmatizer:
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()


def preprocess_text(input_text):
    # print(f'Original text {input_text}')
    sentence = str(input_text)

    cleantext = sentence.replace('{html}', "")
    cleantext = re.sub('<[^<]+?>', '', cleantext)  # remove html tags
    cleantext = re.sub(r'http\S+', '', cleantext)  # remove urls

    if False:
        cleantext = sentence.lower()
        cleantext = re.sub('%?', '', cleantext)  # remove percentage char
        cleantext = re.sub('[0-9]+', '', cleantext)  # remove numbers
        # print(f'Cleaned text {cleantext}')

    if not disable_lemmatizer:
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(cleantext)
        filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
        filtered_text = " ".join(filtered_words)

        stem_words = [stemmer.stem(w) for w in filtered_words]
        lemma_words = [lemmatizer.lemmatize(w) for w in stem_words]
        transformed_text = " ".join(lemma_words)
    else:
        transformed_text = cleantext

    #print(f'*Transformed text* \n {transformed_text}')

    return transformed_text
