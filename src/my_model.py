import re

import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.preprocessing.text import Tokenizer
from keras.src.utils import to_categorical, pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

from src import utils

path_test_txt = 'data/test.txt'
path_train_txt = 'data/train.txt'
path_val_txt = 'data/val.txt'

path_model_saved = 'models/t2es_rnn'


class MyModel:
    stopwords: set[str]
    lemmatizer: WordNetLemmatizer

    tokenizer: Tokenizer
    le: LabelEncoder
    model: None

    def __init__(self):
        # region download necessary packages
        nltk.download("stopwords")
        self.stop_words = set(stopwords.words("english"))

        self.lemmatizer = WordNetLemmatizer()

        nltk.download('wordnet')
        nltk.download('omw-1.4')

        # endregion

        # region load datasets
        df_train = pd.read_csv(path_train_txt, names=['Text', 'Emotion'], sep=';')
        df_val = pd.read_csv(path_val_txt, names=['Text', 'Emotion'], sep=';')
        df_test = pd.read_csv(path_test_txt, names=['Text', 'Emotion'], sep=';')

        utils.try_rm_dupl_df(df_train)
        utils.try_rm_dupl_df(df_val)
        utils.try_rm_dupl_df(df_test)

        # endregion

        df_train = self.normalize_text(df_train)
        df_test = self.normalize_text(df_test)
        df_val = self.normalize_text(df_val)

        # Splitting the text from the labels
        X_train = df_train['Text']
        y_train = df_train['Emotion']

        X_test = df_test['Text']
        y_test = df_test['Emotion']

        X_val = df_val['Text']
        y_val = df_val['Emotion']

        # Encode labels
        self.le = LabelEncoder()
        y_train = self.le.fit_transform(y_train)
        y_test = self.le.transform(y_test)
        y_val = self.le.transform(y_val)

        # Convert the class vector (integers) to binary class matrix
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        y_val = to_categorical(y_val)

        # Tokenize words
        self.tokenizer = Tokenizer(oov_token='UNK')
        self.tokenizer.fit_on_texts(pd.concat([X_train, X_test], axis=0))

        maxlen = max([len(t) for t in df_train['Text']])

        # sdfasdf
        sequences_train = self.tokenizer.texts_to_sequences(X_train)
        sequences_test = self.tokenizer.texts_to_sequences(X_test)
        sequences_val = self.tokenizer.texts_to_sequences(X_val)

        X_train = pad_sequences(sequences_train, maxlen=229, truncating='pre')
        X_test = pad_sequences(sequences_test, maxlen=229, truncating='pre')
        X_val = pad_sequences(sequences_val, maxlen=229, truncating='pre')

        vocabSize = len(self.tokenizer.index_word) + 1

        # predict model
        self.model = tf.keras.models.load_model(path_model_saved)

        print('model is ready !!!')

    def predict(self, sentence: str) -> tuple[str, float]:
        print(sentence)
        sentence = self.normalized_sentence(sentence)
        sentence = self.tokenizer.texts_to_sequences([sentence])
        sentence = pad_sequences(sentence, maxlen=229, truncating='pre')

        d1 = self.model.predict(sentence)
        d2 = np.argmax(d1, axis=-1)
        d3 = self.le.inverse_transform(d2)
        result = d3[0]
        proba = np.max(self.model.predict(sentence))

        return result, float(proba)

    def predicts(self, sentence: str) -> list[tuple[str, float]]:
        print(sentence)
        sentence = self.normalized_sentence(sentence)
        sentence = self.tokenizer.texts_to_sequences([sentence])
        sentence = pad_sequences(sentence, maxlen=229, truncating='pre')
        predictions = self.model.predict(sentence)[0]
        results = []
        for index, prediction in enumerate(predictions):
            result = self.le.inverse_transform(np.array([index]))[0]
            results.append((result, float(prediction)))

        return results

    # region functions

    @staticmethod
    def lemmatization(text):
        lemmatizer = WordNetLemmatizer()

        text = text.split()

        text = [lemmatizer.lemmatize(y) for y in text]

        return " ".join(text)

    def remove_stop_words(self, text):
        Text = [i for i in str(text).split() if i not in self.stop_words]
        return " ".join(Text)

    @staticmethod
    def Removing_numbers(text):
        text = ''.join([i for i in text if not i.isdigit()])
        return text

    @staticmethod
    def lower_case(text):
        text = text.split()

        text = [y.lower() for y in text]

        return " ".join(text)

    @staticmethod
    def Removing_punctuations(text):
        # Remove punctuations
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "", )

        # remove extra whitespace
        text = re.sub('\s+', ' ', text)
        text = " ".join(text.split())
        return text.strip()

    @staticmethod
    def Removing_urls(text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    @staticmethod
    def remove_small_sentences(df):
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan

    def normalized_sentence(self, sentence):
        sentence = self.lower_case(sentence)
        sentence = self.remove_stop_words(sentence)
        sentence = self.Removing_numbers(sentence)
        sentence = self.Removing_punctuations(sentence)
        sentence = self.Removing_urls(sentence)
        sentence = self.lemmatization(sentence)
        return sentence

    def normalize_text(self, df):
        df.Text = df.Text.apply(lambda text: self.lower_case(text))
        df.Text = df.Text.apply(lambda text: self.remove_stop_words(text))
        df.Text = df.Text.apply(lambda text: self.Removing_numbers(text))
        df.Text = df.Text.apply(lambda text: self.Removing_punctuations(text))
        df.Text = df.Text.apply(lambda text: self.Removing_urls(text))
        df.Text = df.Text.apply(lambda text: self.lemmatization(text))
        return df

    # endregion
