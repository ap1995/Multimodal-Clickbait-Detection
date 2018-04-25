import numpy as np
import pandas as pd
from collections import defaultdict
import re
import jsonlines

import sys
import os

os.environ['KERAS_BACKEND'] = 'theano'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model, load_model
from sklearn.metrics import accuracy_score

def run(train_data, test_data, truth_data):
    textFeatures = ["postText", "targetCaptions", "targetParagraphs", "targetTitle", "targetKeywords",
                    "targetDescription"]

    train_data_df = pd.DataFrame.from_dict(train_data)
    truth_data_df = pd.DataFrame.from_dict(truth_data)
    train = pd.merge(train_data_df, truth_data_df, on="id")
    data = train[textFeatures].values #TRAIN + VALID
    print("Data shape ", data.shape)

    test_data_df = pd.DataFrame.from_dict(test_data)
    test = pd.merge(test_data_df, truth_data_df, on="id")
    tdata = test[textFeatures].values

    labels = []
    tlabels = []
    df = []

    # for i in train.values:
    #     if(i[9]=="clickbait"):
    #         labels.append(1)
    #     else:
    #         labels.append(0)

    for i in test.values:
        if (i[9] == "clickbait"):
            tlabels.append(1)
        else:
            tlabels.append(0)

    for i in range(data.shape[0]):
        text = []
        for j in range(0,5):
            k = data[i][j]
            if (j == 3 or j == 4):
                text.append(k)
            else:
                text+=k
        words = ""
        for string in text:
            string = clean_str(string)
            words +=" ".join(string.split())
        df+=[words]

    t_df = []
    for i in range(tdata.shape[0]):
        text = []
        for j in range(0,5):
            k = tdata[i][j]
            if (j == 3 or j == 4):
                text.append(k)
            else:
                text+=k
        words = ""
        for string in text:
            string = clean_str(string)
            words +=" ".join(string.split())
        t_df+=[words]

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(df)
    # sequences = tokenizer.texts_to_sequences(df)

    tokenizer_test = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer_test.fit_on_texts(t_df)
    test_sequences = tokenizer_test.texts_to_sequences(t_df)

    word_index = tokenizer.word_index
    # print('Found %s unique tokens.' % len(word_index))

    # data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    tdata = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # labels = to_categorical(np.asarray(labels))
    tlabels = to_categorical(np.asarray(tlabels))
    # print('Shape of data tensor:', data.shape)
    # print('Shape of label tensor:', labels.shape)
    #
    # indices = np.arange(data.shape[0])
    # np.random.shuffle(indices)
    # data = data[indices]
    # labels = labels[indices]
    # nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    # x_train = data[:-nb_validation_samples]
    # y_train = labels[:-nb_validation_samples]
    # x_val = data[-nb_validation_samples:]
    # y_val = labels[-nb_validation_samples:]
    x_test = tdata
    y_test = tlabels
    print("X_test ", x_test.shape)

    # print('Training and validation sets')
    # print(y_train.sum(axis=0))
    # print(y_val.sum(axis=0))

    embeddings_index = {}
    f = open('glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
    preds = Dense(2, activation='softmax')(l_lstm)

    model = Model(sequence_input, preds)
    model.load_weights('weights-improvement-03-0.80.hdf5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    print("model fitting - Bidirectional LSTM")
    model.summary()
    preds = model.predict(x_test, batch_size=50, verbose=1)
    preds = preds.round()
    print(accuracy_score(y_test, preds))
    # model.evaluate(x_test, y_test, batch_size=50) ################## ADDED LINE


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"@", "", string)
    return string.lower()


MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1111806

count = 0
train_val_data = []
test_data = []
with jsonlines.open('instances.jsonl') as reader:
    for obj in reader.iter(type=dict, skip_invalid=True):
        count += 1
        if (count > 17584):
            test_data.append(obj)
        if(count<=17584):
            train_val_data.append(obj)

count = 0
truth_data = []
with jsonlines.open('truth.jsonl') as reader:
    for obj in reader.iter(type=dict, skip_invalid=True):
        truth_data.append(obj)

run(train_val_data, test_data, truth_data)