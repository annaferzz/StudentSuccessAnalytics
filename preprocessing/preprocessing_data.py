import pandas as pd
import essay
import pickle
import re


def mbti_to_big5(mbti):
    """
        Преобразует тип личности MBTI в черты Big Five.

        :param mbti: Тип личности MBTI.
        :return: Кортеж из пяти черт Big Five.
        """
    mbti = mbti.lower()
    cEXT, cNEU, cAGR, cCON, cOPN = 0, 0, 0, 0, 0

    if mbti[0] == "i":
        cEXT = 0
    elif mbti[0] == "e":
        cEXT = 1

    if mbti[1] == "n":
        cOPN = 1
    elif mbti[1] == "s":
        cOPN = 0

    if mbti[2] == "t":
        cAGR = 0
    elif mbti[2] == "f":
        cAGR = 1

    if mbti[3] == "p":
        cCON = 0
    elif mbti[3] == "j":
        cCON = 1

    return cEXT, cNEU, cAGR, cCON, cOPN


def remove_unemotional_scentences(emotional_words, text_as_one_string):
    """
        Удаляет нейтральные предложения из текста.

        :param emotional_words: Список эмоциональных слов.
        :param text_as_one_string: Текст в виде одной строки.
        :return: Отфильтрованный текст.
        """
    reduced_s = ""
    scentences = re.split('(?<=[.!?]) +', text_as_one_string)
    for s in scentences:
        if any(e in s for e in emotional_words):
            reduced_s = reduced_s + s + " "
        else:
            pass
    return reduced_s


def create_essays(df, subtract=None):
    """
        Создает список на основе DataFrame.

        :param df: DataFrame с данными.
        :param subtract: Список слов для фильтрации.
        :return: Список объектов класса Essay.
        """
    essays = []
    for index, row in df.iterrows():
        essays.append(essay.Essay(row.TEXT, row.cEXT, row.cNEU, row.cAGR, row.cCON, row.cOPN))

    if subtract is not None:
        for x in essays:
            x.filtered_text = remove_unemotional_scentences(emotional_words, x.clean_text)

    return essays


# Загрузка определений
df_essays = pd.read_csv('../data/training/essays.csv', encoding='cp1252', delimiter=',', quotechar='"')
# Для каждого определения мы заменяем категории личности
for e in df_essays.columns[2:7]:
    df_essays[e] = df_essays[e].replace('n', '0')
    df_essays[e] = df_essays[e].replace('y', '1')
    df_essays[e] = pd.to_numeric(df_essays[e])

df_essays = df_essays[["TEXT", "cEXT", "cNEU", "cAGR", "cCON", "cOPN"]]
print(df_essays)

# Загрузка MBTI kaggle dataset
df_kaggle = pd.read_csv('../data/training/mbti_1.csv', skiprows=0)

df_kaggle["cEXT"] = df_kaggle.apply(lambda x: mbti_to_big5(x.type)[0], 1)
df_kaggle["cNEU"] = df_kaggle.apply(lambda x: mbti_to_big5(x.type)[1], 1)
df_kaggle["cAGR"] = df_kaggle.apply(lambda x: mbti_to_big5(x.type)[2], 1)
df_kaggle["cCON"] = df_kaggle.apply(lambda x: mbti_to_big5(x.type)[3], 1)
df_kaggle["cOPN"] = df_kaggle.apply(lambda x: mbti_to_big5(x.type)[4], 1)

df_kaggle = df_kaggle[["posts", "cEXT", "cNEU", "cAGR", "cCON", "cOPN"]]
df_kaggle.columns = ["TEXT", "cEXT", "cNEU", "cAGR", "cCON", "cOPN"]

df_kaggle["TEXT"] = df_kaggle.apply(lambda x: x.TEXT.replace("|||", " ")[:], 1)

print(df_kaggle)

# Загрузка Emotional Lexicon
df_lexicon = pd.read_csv('../data/training/Emotion_Lexicon.csv', index_col=0)
df_lexicon = df_lexicon[(df_lexicon.T != 0).any()]
emotional_words = df_lexicon.index.tolist()

# Объединение наборов данных
essays = create_essays(df_essays, emotional_words)
data = len(essays)
filename = "data/essays/essays" + str(data) + ".pickle"
pickle.dump(essays, open(filename, "wb"))
print("saved entries: ", len(essays))

frames = [df_essays, df_kaggle]
essays_kaggle = pd.concat(frames, sort=False)
essays_kaggle.reset_index(drop=True)

essays_kaggle = create_essays(essays_kaggle, emotional_words)
data = len(essays_kaggle)
filename = "data/essays/essays" + str(data) + ".pickle"
pickle.dump(essays_kaggle, open(filename, "wb"))
print("saved entries: ", len(essays_kaggle))

# Проверка правильности загрузки данных
with open("../data/essays/essays2467.pickle", "rb") as f:
    essays = pickle.load(f)

df_essays = pd.DataFrame([e.__dict__ for e in essays])
print(df_essays.head())

with open("../data/essays/essays11142.pickle", "rb") as f:
    essays_kaggle = pickle.load(f)

df_essays_kaggle = pd.DataFrame([e.__dict__ for e in essays_kaggle])
print(df_essays_kaggle.head())
