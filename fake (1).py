# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 14:18:32 2018

@author: sandeep
"""

import keras
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
import numpy as np
import pandas as pd
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import re


import sys
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import string
from matplotlib import pyplot as plt
from keras.layers import Dense, Embedding, LSTM, GRU

MAX_SEQUENCE_LENGTH = 300
MAX_NB_WORDS = 50000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

data_train = pd.read_csv('train.csv')

texts = []
labels = []
stop_words = set(stopwords.words('english'))

for i in range(data_train.text.shape[0]):
    text1 = data_train.title[i]
    text2 = data_train.text[i]
    text = str(text1) +""+ str(text2)
    texts.append(text)
    labels.append(data_train.label[i])
    
tokenizer1 = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer1.fit_on_texts(texts)
sequences1 = tokenizer1.texts_to_sequences(texts)
word_index1 = tokenizer1.word_index

text2= []
for i in texts:

    word_tokens = word_tokenize(i)

    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    filtered_sentence = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    filtered_sentence = "".join([" "+j if not j.startswith("'") and j not in string.punctuation else j for j in filtered_sentence]).strip()
    text2.append(filtered_sentence)
    

    

    #print(word_tokens)
    #print(filtered_sentence)



    


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(text2)
sequences = tokenizer.texts_to_sequences(text2)
word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels),num_classes = 2)


#Using Pre-trained word embeddings
GLOVE_DIR = "embeddings" 
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf8")
for line in f:
    values = line.split()
    #print(values[1:])
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

num_words = min(MAX_NB_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

# Train test validation Split
from sklearn.model_selection import train_test_split

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train, x_test, y_train, y_test = train_test_split( data, labels, test_size=0.20, random_state=42)
#x_test, x_val, y_test, y_val = train_test_split( x_test, y_test, test_size=0.50, random_state=42)
print('Size of train, validation, test:', len(y_train), len(y_test))

print('real & fake news in train,valt,test:')
print(y_train.sum(axis=0))
print(y_test.sum(axis=0))



modell = Sequential()
modell.add(embedding_layer)

modell.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
modell.add(MaxPooling1D(pool_size=2))
modell.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
modell.add(MaxPooling1D(pool_size=2))
modell.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
modell.add(BatchNormalization())
modell.add(Dense(256, activation='relu'))
modell.add(Dense(128, activation='relu'))
modell.add(Dropout(0.2))
modell.add(Dense(64, activation='relu'))
modell.add(Dense(2, activation='softmax'))

modell.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(modell.summary())
modell.fit(x_train, y_train, epochs=15, batch_size=128)
y_pred = modell.predict(x_test)
y_pred1 = modell.predict(x_test)


y_pred = y_pred[:,1:]
for i in range(0, 4160):
    if y_pred[i]>0.5:
        y_pred[i]=1.0
    else:
        y_pred[i]=0.0


from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)

modell.save('lstm.h5')
