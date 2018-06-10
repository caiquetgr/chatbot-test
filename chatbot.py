# data manipulation
import pandas as pd
# numeric array manipulation
import numpy as np

# natural language toolkit
import nltk

#__utils class
from chatbot_utils import ChatbotUtil

# machine learning models
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# predict utils
from collections import Counter

class ChatBot:
    def __init__(self):
        self.__clfNB = MultinomialNB()
        self.__clfForest = RandomForestClassifier(n_estimators=10, max_depth=5)
        self.__clfNeural = MLPClassifier(activation='relu')
        self.__stemmer = nltk.stem.RSLPStemmer()
        
        self.__isTrained = False

        self.__utils = ChatbotUtil()
        self.__words_encoded = None
        self.__answers_encoded = None
        self.df = pd.read_csv('./chatbot_respostas.csv')

    def train(self, df, questionColumn, answerColumn):
        questions = df[questionColumn]
        answers = df[answerColumn]

        dictionary = self.__utils.prepare_dictionary(questions)
        total_words = len(dictionary)

        tuples = zip(dictionary, range(total_words))
        self.__words_encoded = {word: index for word, index in tuples}

        self.__answers_encoded = self.__utils.encode_answers(answers)

        X = []
        y = []

        for question, answer in df.values:
            X.append(self.__utils.encode_text(question, self.__words_encoded))
            y.append(self.__answers_encoded[answer])

        print(self.__answers_encoded)

        X = np.array(X)
        y = np.array(y)

        self.__clfNB.fit(X, y)
        self.__clfForest.fit(X, y)
        self.__clfNeural.fit(X, y)

        self.__isTrained = True


    def answer(self, question):
        if not self.__isTrained:
            raise RuntimeError('Chatbot não foi treinado.')

        text_encoded = [self.__utils.encode_text(question, self.__words_encoded)]

        pred_nb = self.__clfNB.predict_proba(text_encoded)
        pred_forest = self.__clfForest.predict_proba(text_encoded)
        pred_ada = self.__clfNeural.predict_proba(text_encoded)

        final_result = ((pred_nb + pred_forest + pred_ada)/3 * 100)[0]

        #print(final_result)

        answer_id = np.argmax(final_result)

        #print(answer_id)

        return answer_id

    def full_answer(self, answer_id, full_answer_column):
        answer_topic = [topic for topic, id in self.__answers_encoded.items() if id == answer_id]

        full_answer = self.df.loc[ self.df['Assunto'] == answer_topic[0] ]

        return full_answer.values[0][1]
