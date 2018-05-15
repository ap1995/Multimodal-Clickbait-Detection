import numpy as np
import pandas as pd
import re
import jsonlines
import os

os.environ['KERAS_BACKEND'] = 'theano'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers.merge import concatenate

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model

import tensorflow as tf
from config import Config
from cnn_model import cnn_model

def run(train_data, test_data, truth_data):
    final_vals = []
    data_df = pd.DataFrame.from_dict(train_data)
    truth_data_df = pd.DataFrame.from_dict(truth_data)
    train = pd.merge(data_df, truth_data_df, on="id")
    features = ["postText", "targetCaptions", "targetParagraphs", "targetTitle", "targetKeywords",
                    "targetDescription", "truthClass"]
    vals = train[features]
    vals = vals.values.tolist()
    # print(vals[0][2])
    for i in range(len(vals)):
        if vals[i][1] != []:
            final_vals.append([vals[i][2], vals[i][4], vals[i][5], vals[i][6], vals[i][7], vals[i][8], vals[i][9]])

    # vals_df = pd.DataFrame(final_vals, columns=features)

    # finalTestvals = []
    # test_data_df = pd.DataFrame.from_dict(test_data)
    # test = pd.merge(test_data_df, truth_data_df, on="id")
    # test_vals = test[features].values.tolist()
    VALIDATION_SPLIT = 0.1
    nb_validation_samples = int(VALIDATION_SPLIT * len(final_vals))
    x_val = final_vals[:nb_validation_samples]
    # x_val_df = pd.DataFrame(x_val, columns=features)
    x_test = final_vals[int(0.8 * len(final_vals)):int(0.9 * len(final_vals))]
    final_vals = final_vals[0:int(len(final_vals) * 0.8)]

    # for i in range(len(x_test)):
    #     if test_data[i][1] != []:
    #         finalTestvals.append([test_data[i][0], [test_data[i][1][0]], [test_data[i][2]], test_data[i][3]])
    # # tdata = test[features].values
    # # tdata = test_data_df.values
    tdata = x_test
    #
    # labels = []
    # tlabels = []
    # titles_train_df = []
    # content_train_df = []
    # val_labels = []

    labels = get_labels(final_vals)
    val_labels = get_labels(x_val)
    tlabels = get_labels(tdata)

    titles_train_df = get_title_df(final_vals)
    content_train_df = get_content_df(final_vals)
    titles_val_df = get_title_df(x_val)
    content_val_df = get_content_df(x_val)

    title_train_data, t_word_index = get_padded_sequences(titles_train_df)
    content_train_data, c_word_index = get_padded_sequences(content_train_df)

    title_val_data = get_padded_sequences(titles_val_df)
    content_val_data = get_padded_sequences(content_val_df)

    labels = to_categorical(np.asarray(labels))
    print('Shape of title data tensor:', title_train_data.shape)
    print('Shape of content data tensor:', content_train_data.shape)
    print('Shape of label tensor:', labels.shape)

    # tindices = np.arange(title_train_data.shape[0])
    # np.random.shuffle(tindices)
    #
    # cindices = np.arange(title_train_data.shape[0])
    # np.random.shuffle(cindices)
    # data = data[indices]
    # labels = labels[indices]
    # nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    # x_train = data[:-nb_validation_samples]
    x_train_title = title_train_data
    x_train_content = content_train_data
    y_train = labels
    # x_val = data[-nb_validation_samples:]
    x_val_title = title_val_data
    x_val_content = content_val_data
    y_val = np.asarray(val_labels)
    # x_test = tdata
    y_test = np.asarray(tlabels)

    print('Training and validation sets')
    print(y_train.sum(axis=0))
    print(y_val.sum(axis=0))

    embeddings_index = {}
    f=open('glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors.' % len(embeddings_index))

    t_embedding_matrix = np.random.random((len(t_word_index) + 1, EMBEDDING_DIM)) ##Titles
    for word, i in t_word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            t_embedding_matrix[i] = embedding_vector

    c_embedding_matrix = np.random.random((len(c_word_index) + 1, EMBEDDING_DIM)) ##Content
    for word, i in c_word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            c_embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(t_word_index) + 1, EMBEDDING_DIM, weights=[t_embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    content_embedding_layer = Embedding(len(c_word_index) + 1, EMBEDDING_DIM, weights=[c_embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
    content_data_input = Input(shape=(MAX_SEQUENCE_LENGTH,),dtype='float32')

    embedded_sequences = embedding_layer(sequence_input)
    l_lstm = Bidirectional(LSTM(100))(embedded_sequences)

    content_embedded_sequences = content_embedding_layer(content_data_input)
    l_lstm_content = Bidirectional(LSTM(100))(content_embedded_sequences)

    print("-----------------------------")
    print(l_lstm.shape)

    preds_title = Dense(2, activation='softmax')(l_lstm)

    preds_content = Dense(2,activation='softmax')(l_lstm_content)

    preds_add = concatenate([preds_title, preds_content], axis =-1)

    preds = Dense(2)(preds_add)

    model = Model([sequence_input, content_data_input], preds)
    # model1.add_update(ac.AttentionWithContext()) ###############

    checkpoint = ModelCheckpoint("weights-text-{epoch:02d}-{val_acc:.2f}.hdf5")
    callbacks_list = [checkpoint]
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    print("model fitting - Bidirectional LSTM with titles and content")
    model.summary()
    print('------')

    model.fit([x_train_title, x_train_content], y_train, validation_data=([x_val_title, x_val_content], y_val), epochs=10, batch_size=50, callbacks=callbacks_list)

def get_labels(vals):
    labels = []
    for i in vals:
        if(i[6]=="clickbait"):
            labels.append(1)
        else:
            labels.append(0)
    return labels

def get_title_df(vals):
    titles_df = []
    for i in range(len(vals)): ## For titles
        text = []
        k = vals[i][1]
        text+=(k)
        words = ""
        for string in text:
            string = clean_str(string)
            words +=" ".join(string.split())
        titles_df+=[words]
    return titles_df

def get_content_df(vals):
    content_df = []
    for i in range(len(vals)): ## For content
        text = []
        for j in range(2, 5):
            k = vals[i][j]
            if(j==3):
                text.append(k)
            else:
                text += (k)
        words = ""
        for string in text:
            string = clean_str(string)
            words += " ".join(string.split())
        content_df += [words]
    return content_df

def get_padded_sequences(df):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(df)
    sequences = tokenizer.texts_to_sequences(df)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return data, word_index

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
# VALIDATION_SPLIT = 0.1111806

count = 0
train_val_data = []
test_data = []

with jsonlines.open('instances.jsonl') as reader:
    for obj in reader.iter(type=dict, skip_invalid=True):
        count += 1
        train_val_data.append(obj)

count = 0
truth_data = []
with jsonlines.open('truth.jsonl') as reader:
    for obj in reader.iter(type=dict, skip_invalid=True):
        truth_data.append(obj)

run(train_val_data, test_data, truth_data)
