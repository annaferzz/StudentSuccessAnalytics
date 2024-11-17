import pickle

import dill as dill
import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Модель, основанная на извлечении признаков GloVe

my_path = os.path.abspath(os.path.dirname('glove.6B.300d.txt'))
GLOVE_BIG = os.path.join(my_path, "../data/pretrained/glove.6B/glove.6B.300d.txt")
GLOVE_SMALL = os.path.join(my_path, "../data/pretrained/glove.6B/glove.6B.50d.txt")
encoding = "utf-8"

# Загружаем предварительно обработанные данные, которые мы сохранили
# Выбираем, сколько данных загрузить (2467, 11142)

# essays = pickle.load(open("data/essays/essays2467.pickle", "rb"))
essays = pickle.load(open("../data/essays/essays11142.pickle", "rb"))
print("loaded count of essays:", len(essays))


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(glove_mywords))])
        else:
            self.dim = 0

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


df = pd.read_csv(GLOVE_SMALL, sep=" ", quoting=3, header=None)
corpus = []

for e in essays:
    for w in e.words:
        corpus.append(w)

df_corpus = pd.DataFrame(corpus)
df_mywords = df.merge(df_corpus)
df_mywords = df_mywords.drop_duplicates()
df_temp = df_mywords.set_index(0)
glove_mywords = {key: val.values for key, val in df_temp.T.items()}

for e in essays:
    df_temp_e = pd.DataFrame(e.words)
    try:
        df_temp_e = df_temp_e.merge(df_mywords)
        df_temp_e = df_temp_e.drop_duplicates()
        df_temp_e = df_temp_e.set_index(0)
        e.glove = {key: val.values for key, val in df_temp_e.T.items()}
    except:
        print("error")

filename = "data/essays/essays_glove" + "50" + "d_" + str(len(essays)) + ".pickle"
pickle.dump(essays, open(filename, "wb"))
print("saved", len(essays), "entries: in", filename)

# Разделение данных на обучающую и тестовую выборки
training, test = train_test_split(essays, test_size=0.20, random_state=42)

train_x = [x.glove for x in training]
train_y_cEXT = [x.cEXT for x in training]
train_y_cNEU = [x.cNEU for x in training]
train_y_cAGR = [x.cAGR for x in training]
train_y_cCON = [x.cCON for x in training]
train_y_cOPN = [x.cOPN for x in training]

test_x = [x.glove for x in test]
test_y_cEXT = [x.cEXT for x in test]
test_y_cNEU = [x.cNEU for x in test]
test_y_cAGR = [x.cAGR for x in test]
test_y_cCON = [x.cCON for x in test]
test_y_cOPN = [x.cOPN for x in test]

train_x = np.array(train_x)
train_y_cEXT = np.array(train_y_cEXT)
train_y_cNEU = np.array(train_y_cNEU)
train_y_cAGR = np.array(train_y_cAGR)
train_y_cCON = np.array(train_y_cCON)
train_y_cOPN = np.array(train_y_cOPN)

glove_vectorizer = MeanEmbeddingVectorizer(glove_mywords)
train_x_vectors = glove_vectorizer.transform(train_x)

test_x_vectors = glove_vectorizer.transform(test_x)

vectorizer_filename = "../data/models/vectorizer_glove.pickle"
with open(vectorizer_filename, 'wb') as f:
    dill.dump(glove_vectorizer, f)

evaluation = []
data = len(essays)
vec_name = "GloVe"

models_path = "../data/models"
if not os.path.exists(models_path):
    os.makedirs(models_path)

# SVM
name = "svm"
print("training Extraversion cEXT using SVM...")
clf_svm_cEXT = svm.SVC(kernel='linear')
clf_svm_cEXT.fit(train_x_vectors, train_y_cEXT)
evaluation.append([data, vec_name, name, "cEXT", clf_svm_cEXT.score(test_x_vectors, test_y_cEXT)])
print("cEXT score: ", clf_svm_cEXT.score(test_x_vectors, test_y_cEXT))

try:
    print("training Neuroticism cNEU using SVM...")
    clf_svm_cNEU = svm.SVC(kernel='linear')
    clf_svm_cNEU.fit(train_x_vectors, train_y_cNEU)
    evaluation.append([data, vec_name, name, "cNEU", clf_svm_cNEU.score(test_x_vectors, test_y_cNEU)])
    print("cNEU score: ", clf_svm_cNEU.score(test_x_vectors, test_y_cNEU))
except:
    print("with this data not available (MBTI only 4 dimensions)")

print("training Agreeableness cAGR using using SVM...")
clf_svm_cAGR = svm.SVC(kernel='linear')
clf_svm_cAGR.fit(train_x_vectors, train_y_cAGR)
evaluation.append([data, vec_name, name, "cAGR", clf_svm_cAGR.score(test_x_vectors, test_y_cAGR)])
print("cAGR score: ", clf_svm_cAGR.score(test_x_vectors, test_y_cAGR))

print("training Conscientiousness cCON using SVM...")
clf_svm_cCON = svm.SVC(kernel='linear')
clf_svm_cCON.fit(train_x_vectors, train_y_cCON)
evaluation.append([data, vec_name, name, "cCON", clf_svm_cCON.score(test_x_vectors, test_y_cCON)])
print("cCON score: ", clf_svm_cCON.score(test_x_vectors, test_y_cCON))

print("training Openness to Experience cOPN using SVM...")
clf_svm_cOPN = svm.SVC(kernel='linear')
clf_svm_cOPN.fit(train_x_vectors, train_y_cOPN)
evaluation.append([data, vec_name, name, "cOPN", clf_svm_cOPN.score(test_x_vectors, test_y_cOPN)])
print("cOPN score: ", clf_svm_cOPN.score(test_x_vectors, test_y_cOPN))

# Decision Tree
name = "tree"
print("training Extraversion cEXT using Decision Tree...")
clf_dec_cEXT = tree.DecisionTreeClassifier()
clf_dec_cEXT.fit(train_x_vectors, train_y_cEXT)
evaluation.append([data, vec_name, name, "cEXT", clf_dec_cEXT.score(test_x_vectors, test_y_cEXT)])
print("cEXT score: ", clf_dec_cEXT.score(test_x_vectors, test_y_cEXT))

try:
    print("training Neuroticism cNEU using Decision Tree...")
    clf_dec_cNEU = tree.DecisionTreeClassifier()
    clf_dec_cNEU.fit(train_x_vectors, train_y_cNEU)
    evaluation.append([data, vec_name, name, "cNEU", clf_dec_cNEU.score(test_x_vectors, test_y_cNEU)])
    print("cNEU score: ", clf_dec_cNEU.score(test_x_vectors, test_y_cNEU))
except:
    print("with this data not available (MBTI only 4 dimensions)")

print("training Agreeableness cAGR using using Decision Tree...")
clf_dec_cAGR = tree.DecisionTreeClassifier()
clf_dec_cAGR.fit(train_x_vectors, train_y_cAGR)
evaluation.append([data, vec_name, name, "cAGR", clf_dec_cAGR.score(test_x_vectors, test_y_cAGR)])
print("cAGR score: ", clf_dec_cAGR.score(test_x_vectors, test_y_cAGR))

print("training Conscientiousness cCON using Decision Tree...")
clf_dec_cCON = tree.DecisionTreeClassifier()
clf_dec_cCON.fit(train_x_vectors, train_y_cCON)
evaluation.append([data, vec_name, name, "cCON", clf_dec_cCON.score(test_x_vectors, test_y_cCON)])
print("cCON score: ", clf_dec_cCON.score(test_x_vectors, test_y_cCON))

print("training Openness to Experience cOPN using Decision Tree...")
clf_dec_cOPN = tree.DecisionTreeClassifier()
clf_dec_cOPN.fit(train_x_vectors, train_y_cOPN)
evaluation.append([data, vec_name, name, "cOPN", clf_dec_cOPN.score(test_x_vectors, test_y_cOPN)])
print("cOPN score: ", clf_dec_cOPN.score(test_x_vectors, test_y_cOPN))

# Logistic Regression
name = "logistic"
print("training Extraversion cEXT using Logistic Regression...")
clf_log_cEXT = LogisticRegression()
clf_log_cEXT.fit(train_x_vectors, train_y_cEXT)
evaluation.append([data, vec_name, name, "cEXT", clf_log_cEXT.score(test_x_vectors, test_y_cEXT)])
print("cEXT score: ", clf_log_cEXT.score(test_x_vectors, test_y_cEXT))

try:
    print("training Neuroticism cNEU using Logistic Regression...")
    clf_log_cNEU = LogisticRegression()
    clf_log_cNEU.fit(train_x_vectors, train_y_cNEU)
    evaluation.append([data, vec_name, name, "cNEU", clf_log_cNEU.score(test_x_vectors, test_y_cNEU)])
    print("cNEU score: ", clf_log_cNEU.score(test_x_vectors, test_y_cNEU))
    with open('../data/models/cNEU.pickle', 'wb') as f:
        pickle.dump(clf_log_cNEU, f)
except:
    print("with this data not available (MBTI only 4 dimensions)")

print("training Agreeableness cAGR using using Logistic Regression...")
clf_log_cAGR = LogisticRegression()
clf_log_cAGR.fit(train_x_vectors, train_y_cAGR)
evaluation.append([data, vec_name, name, "cAGR", clf_log_cAGR.score(test_x_vectors, test_y_cAGR)])
print("cAGR score: ", clf_log_cAGR.score(test_x_vectors, test_y_cAGR))

print("training Conscientiousness cCON using Logistic Regression...")
clf_log_cCON = LogisticRegression()
clf_log_cCON.fit(train_x_vectors, train_y_cCON)
evaluation.append([data, vec_name, name, "cCON", clf_log_cCON.score(test_x_vectors, test_y_cCON)])
print("cCON score: ", clf_log_cCON.score(test_x_vectors, test_y_cCON))

print("training Openness to Experience cOPN using Logistic Regression...")
clf_log_cOPN = LogisticRegression()
clf_log_cOPN.fit(train_x_vectors, train_y_cOPN)
evaluation.append([data, vec_name, name, "cOPN", clf_log_cOPN.score(test_x_vectors, test_y_cOPN)])
print("cOPN score: ", clf_log_cOPN.score(test_x_vectors, test_y_cOPN))

# Random Forest
name = "forest"
print("training Extraversion cEXT using Random Forest...")
clf_for_cEXT = RandomForestClassifier()
clf_for_cEXT.fit(train_x_vectors, train_y_cEXT)
evaluation.append([data, vec_name, name, "cEXT", clf_for_cEXT.score(test_x_vectors, test_y_cEXT)])
print("cEXT score: ", clf_for_cEXT.score(test_x_vectors, test_y_cEXT))

try:
    print("training Neuroticism cNEU using Random Forest...")
    clf_for_cNEU = RandomForestClassifier()
    clf_for_cNEU.fit(train_x_vectors, train_y_cNEU)
    evaluation.append([data, vec_name, name, "cNEU", clf_for_cNEU.score(test_x_vectors, test_y_cNEU)])
    print("cNEU score: ", clf_for_cNEU.score(test_x_vectors, test_y_cNEU))
except:
    print("with this data not available (MBTI only 4 dimensions)")

print("training Agreeableness cAGR using using Random Forest...")
clf_for_cAGR = RandomForestClassifier()
clf_for_cAGR.fit(train_x_vectors, train_y_cAGR)
evaluation.append([data, vec_name, name, "cAGR", clf_for_cAGR.score(test_x_vectors, test_y_cAGR)])
print("cAGR score: ", clf_for_cAGR.score(test_x_vectors, test_y_cAGR))

print("training Conscientiousness cCON using Random Forest...")
clf_for_cCON = RandomForestClassifier()
clf_for_cCON.fit(train_x_vectors, train_y_cCON)
evaluation.append([data, vec_name, name, "cCON", clf_for_cCON.score(test_x_vectors, test_y_cCON)])
print("cCON score: ", clf_for_cCON.score(test_x_vectors, test_y_cCON))

print("training Openness to Experience cOPN using Random Forest...")
clf_for_cOPN = RandomForestClassifier()
clf_for_cOPN.fit(train_x_vectors, train_y_cOPN)
evaluation.append([data, vec_name, name, "cOPN", clf_for_cOPN.score(test_x_vectors, test_y_cOPN)])
print("cOPN score: ", clf_for_cOPN.score(test_x_vectors, test_y_cOPN))

# Оценка модели Glove
filename = "data/evaluation/evaluation" + str(data) + vec_name + ".pickle"
pickle.dump(evaluation, open(filename, "wb"))
print("evaluation saved as", filename)

print(evaluation)
