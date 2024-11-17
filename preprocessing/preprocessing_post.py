import re
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords

stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()

patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"


def remove_non_alphanumeric(text):
    clean_text = re.sub(r'[^\w\s,.;:!?-]', '', text)
    return clean_text


def lemmatize(doc):
    doc = remove_non_alphanumeric(doc)
    doc = re.sub(patterns, ' ', doc)

    tokens = []
    for token in doc.split():
        if token and token not in stopwords_ru:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            tokens.append(token)

    if len(tokens) > 2:
        return ' '.join(tokens)

    return None