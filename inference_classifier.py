from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import numpy as np

import joblib

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.wordnet import WordNetLemmatizer
from pymorphy2 import MorphAnalyzer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('russian'))
patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
morph = MorphAnalyzer()

def lemmatize(doc):
    doc = re.sub(patterns, ' ', doc)
    tokens = []
    for token in doc.split():
        if token.isalpha() and token not in stop_words:
            token = token.strip()
            token = morph.normal_forms(token)[0]       
            tokens.append(token)
    return tokens

def pre_process(text):
    text=text.lower()
    text=re.sub("&amp;lt;/?.*?&amp;gt;"," &amp;lt;&amp;gt; ",text)
    text=re.sub("(\\d|\\W)+"," ",text)
    # text = [word for word in text if len(word) >= 3]
    text = lemmatize(text)
    return ' '.join(text)

class_id_to_classname = {0: 'Автомобили',
 1: 'Астрология и эзотерика',
 2: 'Бизнес и карьера',
 3: 'Демография и население',
 4: 'Домашние питомцы и животные',
 5: 'Еда и напитки',
 6: 'Здравоохранение',
 7: 'Знаменитости и светская жизнь',
 8: 'Интернет и социальные сети',
 9: 'Искусство и культура',
 10: 'История',
 11: 'Космос и астрономия',
 12: 'Мода и стиль жизни',
 13: 'Наука и технологии',
 14: 'Недвижимость',
 15: 'Образование',
 16: 'Окружающая среда и климат',
 17: 'Политика',
 18: 'Преступность и правосудие',
 19: 'Происшествия',
 20: 'Промышленность',
 21: 'Развлечения',
 22: 'Религия и верования',
 23: 'Сельское хозяйство',
 24: 'Социальные вопросы',
 25: 'Спорт',
 26: 'Транспорт',
 27: 'Туризм и путешествия',
 28: 'Экономика'}

vectorizer = None
with open('vectorizer_1.joblib.pkl', 'rb') as fid:
    vectorizer = joblib.load(fid)

classifier = None
with open('classifier_1.joblib.pkl', 'rb') as fid:
    classifier = joblib.load(fid)

#input = ["В рамках проекта Здоровье нации власти города выделили средства на реконструкцию и модернизацию больниц и поликлиник. Обновление медицинской инфраструктуры направлено на повышение качества и доступности медицинской помощи для всех граждан."]


def classify(input):
    input_preprocessed = [pre_process(x) for x in input]
    label = classifier.predict(vectorizer.transform(input_preprocessed))
    return class_id_to_classname[label[0]]
