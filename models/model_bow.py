import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from time import time
start = time()

# Модель, основанная на извлечении признаков Bags of Words
# Загружаем предварительно обработанные данные, которые мы сохранили
# Выбираем, сколько данных загрузить (2467, 11142)

# essays = pickle.load(open("data/essays/essays2467.pickle", "rb"))
essays = pickle.load(open("../data/essays/essays11142.pickle", "rb"))

print("loaded count of essays:", len(essays))

training, test = train_test_split(essays, test_size=0.20, random_state=42)

train_x = [x.clean_text for x in training]
train_y_cEXT = [x.cEXT for x in training]
train_y_cNEU = [x.cNEU for x in training]
train_y_cAGR = [x.cAGR for x in training]
train_y_cCON = [x.cCON for x in training]
train_y_cOPN = [x.cOPN for x in training]

test_x = [x.clean_text for x in test]
test_y_cEXT = [x.cEXT for x in test]
test_y_cNEU = [x.cNEU for x in test]
test_y_cAGR = [x.cAGR for x in test]
test_y_cCON = [x.cCON for x in test]
test_y_cOPN = [x.cOPN for x in test]

# bags of words
bow_vectorizer = CountVectorizer()

# Создаем векторы из наших слов
train_x_vectors = bow_vectorizer.fit_transform(train_x)
test_x_vectors = bow_vectorizer.transform(test_x)

# Сохранение векторизатора
vectorizer_filename = f"../data/models/vectorizer_bow.pickle"
pickle.dump(bow_vectorizer, open(vectorizer_filename, "wb"))
print("Vectorizer saved as", vectorizer_filename)

evaluation = []
data = len(essays)
vec_name = "BoW"

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
print("training Extraversion cEXT using dec...")
clf_dec_cEXT = tree.DecisionTreeClassifier()
clf_dec_cEXT.fit(train_x_vectors, train_y_cEXT)
evaluation.append([data, vec_name, name, "cEXT", clf_dec_cEXT.score(test_x_vectors, test_y_cEXT)])
print("cEXT score: ", clf_dec_cEXT.score(test_x_vectors, test_y_cEXT))

try:
    print("training Neuroticism cNEU using dec...")
    clf_dec_cNEU = tree.DecisionTreeClassifier()
    clf_dec_cNEU.fit(train_x_vectors, train_y_cNEU)
    evaluation.append([data, vec_name, name, "cNEU", clf_dec_cNEU.score(test_x_vectors, test_y_cNEU)])
    print("cNEU score: ", clf_dec_cNEU.score(test_x_vectors, test_y_cNEU))
except:
    print("with this data not available (MBTI only 4 dimensions)")

print("training Agreeableness cAGR using using dec...")
clf_dec_cAGR = tree.DecisionTreeClassifier()
clf_dec_cAGR.fit(train_x_vectors, train_y_cAGR)
evaluation.append([data, vec_name, name, "cAGR", clf_dec_cAGR.score(test_x_vectors, test_y_cAGR)])
print("cAGR score: ", clf_dec_cAGR.score(test_x_vectors, test_y_cAGR))

print("training Conscientiousness cCON using dec...")
clf_dec_cCON = tree.DecisionTreeClassifier()
clf_dec_cCON.fit(train_x_vectors, train_y_cCON)
evaluation.append([data, vec_name, name, "cCON", clf_dec_cCON.score(test_x_vectors, test_y_cCON)])
print("cCON score: ", clf_dec_cCON.score(test_x_vectors, test_y_cCON))

print("training Openness to Experience cOPN using dec...")
clf_dec_cOPN = tree.DecisionTreeClassifier()
clf_dec_cOPN.fit(train_x_vectors, train_y_cOPN)
evaluation.append([data, vec_name, name, "cOPN", clf_dec_cOPN.score(test_x_vectors, test_y_cOPN)])
print("cOPN score: ", clf_dec_cOPN.score(test_x_vectors, test_y_cOPN))

# Logistic Regression

name = "logR"
print("training Extraversion cEXT using Logistic Regression...")
clf_log_cEXT = LogisticRegression(solver="newton-cg")
clf_log_cEXT.fit(train_x_vectors, train_y_cEXT)
evaluation.append([data, vec_name, name, "cEXT", clf_log_cEXT.score(test_x_vectors, test_y_cEXT)])
print("cEXT score: ", clf_log_cEXT.score(test_x_vectors, test_y_cEXT))

with open('../data/models/cEXT.pickle', 'wb') as f:
    pickle.dump(clf_log_cEXT, f)

try:
    print("training Neuroticism cNEU using Logistic Regression...")
    clf_log_cNEU = LogisticRegression(solver="newton-cg")
    clf_log_cNEU.fit(train_x_vectors, train_y_cNEU)
    evaluation.append([data, vec_name, name, "cNEU", clf_log_cNEU.score(test_x_vectors, test_y_cNEU)])
    print("cNEU score: ", clf_log_cNEU.score(test_x_vectors, test_y_cNEU))
    with open('../data/models/cNEU.pickle', 'wb') as f:
        pickle.dump(clf_log_cNEU, f)
except:
    print("with this data not available (MBTI only 4 dimensions)")

print("training Agreeableness cAGR using using Logistic Regression...")
clf_log_cAGR = LogisticRegression(solver="newton-cg")
clf_log_cAGR.fit(train_x_vectors, train_y_cAGR)
evaluation.append([data, vec_name, name, "cAGR", clf_log_cAGR.score(test_x_vectors, test_y_cAGR)])
print("cAGR score: ", clf_log_cAGR.score(test_x_vectors, test_y_cAGR))
with open('../data/models/cAGR.pickle', 'wb') as f:
    pickle.dump(clf_log_cAGR, f)

print("training Conscientiousness cCON using Logistic Regression...")
clf_log_cCON = LogisticRegression(solver="newton-cg")
clf_log_cCON.fit(train_x_vectors, train_y_cCON)
evaluation.append([data, vec_name, name, "cCON", clf_log_cCON.score(test_x_vectors, test_y_cCON)])
print("cCON score: ", clf_log_cCON.score(test_x_vectors, test_y_cCON))
with open('../data/models/cCON.pickle', 'wb') as f:
    pickle.dump(clf_log_cCON, f)


print("training Openness to Experience cOPN using Logistic Regression...")
clf_log_cOPN = LogisticRegression(solver="newton-cg")
clf_log_cOPN.fit(train_x_vectors, train_y_cOPN)
evaluation.append([data, vec_name, name, "cOPN", clf_log_cOPN.score(test_x_vectors, test_y_cOPN)])
print("cOPN score: ", clf_log_cOPN.score(test_x_vectors, test_y_cOPN))
with open('../data/models/cOPN.pickle', 'wb') as f:
    pickle.dump(clf_log_cOPN, f)

# Random Forest

name = "RF"
print("training Extraversion cEXT using Random Forest...")
clf_rf_cEXT = RandomForestClassifier(n_estimators=100)
clf_rf_cEXT.fit(train_x_vectors, train_y_cEXT)
evaluation.append([data, vec_name, name, "cEXT", clf_rf_cEXT.score(test_x_vectors, test_y_cEXT)])
print("cEXT score: ", clf_rf_cEXT.score(test_x_vectors, test_y_cEXT))

try:
    print("training Neuroticism cNEU using Random Forest...")
    clf_rf_cNEU = RandomForestClassifier(n_estimators=100)
    clf_rf_cNEU.fit(train_x_vectors, train_y_cNEU)
    evaluation.append([data, vec_name, name, "cNEU", clf_rf_cNEU.score(test_x_vectors, test_y_cNEU)])
    print("cNEU score: ", clf_rf_cNEU.score(test_x_vectors, test_y_cNEU))
except:
    print("with this data not available (MBTI only 4 dimensions)")

print("training Agreeableness cAGR using using Random Forest...")
clf_rf_cAGR = RandomForestClassifier(n_estimators=100)
clf_rf_cAGR.fit(train_x_vectors, train_y_cAGR)
evaluation.append([data, vec_name, name, "cAGR", clf_rf_cAGR.score(test_x_vectors, test_y_cAGR)])
print("cAGR score: ", clf_rf_cAGR.score(test_x_vectors, test_y_cAGR))

print("training Conscientiousness cCON using Random Forest...")
clf_rf_cCON = RandomForestClassifier(n_estimators=100)
clf_rf_cCON.fit(train_x_vectors, train_y_cCON)
evaluation.append([data, vec_name, name, "cCON", clf_rf_cCON.score(test_x_vectors, test_y_cCON)])
print("cCON score: ", clf_rf_cCON.score(test_x_vectors, test_y_cCON))

print("training Openness to Experience cOPN using Random Forest...")
clf_rf_cOPN = RandomForestClassifier(n_estimators=100)
clf_rf_cOPN.fit(train_x_vectors, train_y_cOPN)
evaluation.append([data, vec_name, name, "cOPN", clf_rf_cOPN.score(test_x_vectors, test_y_cOPN)])
print("cOPN score: ", clf_rf_cOPN.score(test_x_vectors, test_y_cOPN))

filename = "data/evaluation/evaluation" + str(data) + vec_name + ".pickle"
pickle.dump(evaluation, open(filename, "wb"))
print("evaluation saved as", filename)
print("Time:", time()-start)
