from chatbot import ChatBot
import pandas as pd

'''import nltk

#baixar em models: rslp e punckt
#baixar em corpora: stopwords
nltk.download()

'''

df = pd.read_csv('./chatbot_dataset_treino.csv')

chatbot = ChatBot()

chatbot.train(df, 'Pergunta', 'Assunto')

print("####### CHATBOT Alpha 0.0.1 #######")

while True:
    question = input('Pergunta: ')
    answer_id = chatbot.answer(question)

    full_answer = chatbot.full_answer(answer_id, 'Assunto')

    print('Chatbot: ', full_answer)