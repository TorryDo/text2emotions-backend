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

# region download necessary packages
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

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


# region functions
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()

    text = text.split()

    text = [lemmatizer.lemmatize(y) for y in text]

    return " ".join(text)


def remove_stop_words(text):
    Text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)


def Removing_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text


def lower_case(text):
    text = text.split()

    text = [y.lower() for y in text]

    return " ".join(text)


def Removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()


def Removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan


def normalized_sentence(sentence):
    sentence = lower_case(sentence)
    sentence = remove_stop_words(sentence)
    sentence = Removing_numbers(sentence)
    sentence = Removing_punctuations(sentence)
    sentence = Removing_urls(sentence)
    sentence = lemmatization(sentence)
    return sentence


def normalize_text(df):
    df.Text = df.Text.apply(lambda text: lower_case(text))
    df.Text = df.Text.apply(lambda text: remove_stop_words(text))
    df.Text = df.Text.apply(lambda text: Removing_numbers(text))
    df.Text = df.Text.apply(lambda text: Removing_punctuations(text))
    df.Text = df.Text.apply(lambda text: Removing_urls(text))
    df.Text = df.Text.apply(lambda text: lemmatization(text))
    return df


# endregion


df_train = normalize_text(df_train)
df_test = normalize_text(df_test)
df_val = normalize_text(df_val)

# Splitting the text from the labels
X_train = df_train['Text']
y_train = df_train['Emotion']

X_test = df_test['Text']
y_test = df_test['Emotion']

X_val = df_val['Text']
y_val = df_val['Emotion']

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_val = le.transform(y_val)

# Convert the class vector (integers) to binary class matrix
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

# Tokenize words
tokenizer = Tokenizer(oov_token='UNK')
tokenizer.fit_on_texts(pd.concat([X_train, X_test], axis=0))

maxlen = max([len(t) for t in df_train['Text']])

# sdfasdf
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)
sequences_val = tokenizer.texts_to_sequences(X_val)

X_train = pad_sequences(sequences_train, maxlen=229, truncating='pre')
X_test = pad_sequences(sequences_test, maxlen=229, truncating='pre')
X_val = pad_sequences(sequences_val, maxlen=229, truncating='pre')

vocabSize = len(tokenizer.index_word) + 1

# predict model
model = tf.keras.models.load_model(path_model_saved)


def predict(sentence: str) -> tuple[str, float]:
    print(sentence)
    sentence = normalized_sentence(sentence)
    sentence = tokenizer.texts_to_sequences([sentence])
    sentence = pad_sequences(sentence, maxlen=229, truncating='pre')

    d1 = model.predict(sentence)
    d2 = np.argmax(d1, axis=-1)
    d3 = le.inverse_transform(d2)
    result = d3[0]
    proba = np.max(model.predict(sentence))

    return result, float(proba)


def predicts(sentence: str) -> list[tuple[str, float]]:
    print(sentence)
    sentence = normalized_sentence(sentence)
    sentence = tokenizer.texts_to_sequences([sentence])
    sentence = pad_sequences(sentence, maxlen=229, truncating='pre')
    predictions = model.predict(sentence)[0]
    results = []
    for index, prediction in enumerate(predictions):
        result = le.inverse_transform(np.array([index]))[0]
        results.append((result, float(prediction)))

    return results


print('model is ready !!!')
# rs = predicts("I'm in the bad mood, but it's okay, the weather is nice")
#
# print(rs)
