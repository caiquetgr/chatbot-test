from unicodedata import normalize
import nltk
import numpy as np

class ChatbotUtil:

    def prepare_dictionary(self, questions):

        # stops words do nltk
        stop_words = nltk.corpus.stopwords.words('portuguese')
        # remove os caracteres com acento das stop words
        '''
           Dois pontos (..) não é separado da palavra quando realizado a
           tokenizacao, porém um ponto (.) e três pontos (...) são.
           Por isso no código abaixo, no replace, é trocado '..' por '...'
        '''
        stop_words = [self.__remove_special_chars(w.replace('..', '...')) for w in stop_words]
        # tokeniza a frase
        questions = questions.str.lower()
        phrases = [self.__tokenize_phrase(question) for question in questions]

        stemmer = nltk.RSLPStemmer()

        dicionario = set()

        ''' para cada palavra na frase, retirar o sufixo (se não estiver nas stopwords)
         e adicionar no dicionario '''
        for phrase in phrases:
            valid_words = [stemmer.stem(word) for word in phrase if word not in stop_words and len(word) > 1]
            dicionario.update(valid_words)

        # retorna lista de palavras da frase, que não são stop words
        return dicionario
        
    '''
    Retorna a frase quebrada por palavras (sem pontos/pontuações)
    Returns the tokenized phrase (without punctuation/special characters)
    '''
    def __tokenize_phrase(self, phrase):
        phrase = self.__remove_special_chars(phrase)
        phrase = nltk.tokenize.word_tokenize(phrase)
        # retorna cada palavra que o comprimento é maior que 1 (exclui virgulas, interrogações, etc)
        return [w for w in phrase if len(w) > 1]

    '''
    Retira pontuações dos caracteres
    Replaces special characters for normalized characters (ex: à -> a)
    '''
    def __remove_special_chars(self, phrase):
        return normalize('NFKD', phrase).encode('ASCII','ignore').decode('ASCII')

    def encode_text(self, text, words_encoded):
        vector = [0] * len(words_encoded)
        words = self.__tokenize_phrase(text.replace('..', '...'))

        stemmer = nltk.RSLPStemmer()

        for word in words:
            if len(word) > 0:
                stemmed_word = stemmer.stem(word)
                if stemmed_word in words_encoded:
                    index = words_encoded[stemmed_word]
                    vector[index] += 1
        
        return vector

    def encode_answers(self, answers):
        answers_encoded = {}
        index = 0

        for answer in answers:
            if(answer not in answers_encoded.keys()):
                answers_encoded[answer] = index
                index += 1

        return answers_encoded

if __name__ == '__main__':
    utils = ChatbotUtil()
    #print(utils.prepare_phrase('Oi, tudo bem? Como estão? Carrão'))