import pickle
import re

cEXT = pickle.load(open("./data/models/cEXT.pickle", "rb"))
cNEU = pickle.load(open("./data/models/cNEU.pickle", "rb"))
cAGR = pickle.load(open("./data/models/cAGR.pickle", "rb"))
cCON = pickle.load(open("./data/models/cCON.pickle", "rb"))
cOPN = pickle.load(open("./data/models/cOPN.pickle", "rb"))
vectorizer_31 = pickle.load(open("./data/models/vectorizer_bow.pickle", "rb"))


def predict_personality(text):
    sentences = re.split("(?<=[.!?]) +", text)
    text_vector_31 = vectorizer_31.transform(sentences)
    EXT = cEXT.predict(text_vector_31)
    AGR = cAGR.predict(text_vector_31)
    CON = cCON.predict(text_vector_31)
    OPN = cOPN.predict(text_vector_31)
    return [EXT[0], AGR[0], CON[0], OPN[0]]


with open('predict/processed_posts.txt', 'r', encoding='utf-8') as file:
    text = file.read()


def translate_personality(predictions):
    translated_predictions = []
    for idx, val in enumerate(predictions):
        if idx == 0:
            if val == 0:
                translated_predictions.append("Экстраверсия: уединенный / замкнутый")
            elif val == 1:
                translated_predictions.append("Экстраверсия: общительный / энергичный")
        elif idx == 1:
            if val == 0:
                translated_predictions.append("Доброжелательность: критикующий, осуждающий")
            elif val == 1:
                translated_predictions.append("Доброжелательность: дружелюбный, сострадательный")
        elif idx == 2:
            if val == 0:
                translated_predictions.append("Добросовестность: покладистый / небрежный")
            elif val == 1:
                translated_predictions.append("Добросовестность: эффективный / организованный")
        elif idx == 3:
            if val == 0:
                translated_predictions.append("Открытость опыту: последовательность / осторожность")
            elif val == 1:
                translated_predictions.append("Открытость опыту: изобретательность / любопытство")
    return translated_predictions


def big5_to_mbti(big5):
    mbti = ""

    if big5[0] == 0:
        mbti += "I"
    elif big5[0] == 1:
        mbti += "E"

    if big5[3] == 0:
        mbti += "S"
    elif big5[3] == 1:
        mbti += "N"

    if big5[1] == 0:
        mbti += "T"
    elif big5[1] == 1:
        mbti += "F"

    if big5[2] == 0:
        mbti += "P"
    elif big5[2] == 1:
        mbti += "J"

    return mbti


predictions = predict_personality(text)
print("predicted personality:", predictions)
translated_predictions = translate_personality(predictions)
mbti_type = big5_to_mbti(predictions)
print('\nМодель «Большая пятерка»:')
for prediction in translated_predictions:
    print(prediction)
print('\nТипология Майерс-Бриггс:', mbti_type)