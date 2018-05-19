import numpy as np
import pandas as pd
import re
import jsonlines
import os

os.environ['KERAS_BACKEND'] = 'theano'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input
from keras.layers import Embedding, LSTM, Bidirectional
from keras.models import Model
from sklearn.metrics import accuracy_score

def run(train_data, test_data, truth_data):
    train_data_df = pd.DataFrame.from_dict(train_data)
    truth_data_df = pd.DataFrame.from_dict(truth_data)
    train = pd.merge(train_data_df, truth_data_df, on="id")
    data = train.values

    vals = data.tolist()
    final_vals = []
    for i in range(len(vals)):
        if vals[i][1] != []:
            final_vals.append([vals[i][2], vals[i][4], vals[i][5], vals[i][6], vals[i][7], vals[i][8], vals[i][9]])

    # columns = "postText", "targetCaptions", "targetParagraphs", "targetTitle", "targetKeywords", "targetDescription", "truthClass"

    df = []
    labels = []
    tlabels = []

    test_data = final_vals[int(0.9 * len(final_vals)):]
    final_vals = final_vals[0:int(len(final_vals)*0.9)]

    for i in final_vals:
        if(i[6]=="clickbait"):
            labels.append(1)
        else:
            labels.append(0)

    for i in test_data:
        if(i[6]=="clickbait"):
            tlabels.append(1)
        else:
            tlabels.append(0)

    for i in range(len(final_vals)):
        text = []
        for j in range(0,5):
            k = final_vals[i][j]
            if (j == 2 or j == 3):
                text.append(k)
            else:
                text+=k
        words = ""
        for string in text:
            string = clean_str(string)
            words +=" ".join(string.split())
        df+=[words]

    test_df= []
    for i in range(len(test_data)):
        text = []
        for j in range(0, 5):
            k = test_data[i][j]
            if (j == 2 or j == 3):
                text.append(k)
            else:
                text += k
        words = ""
        for string in text:
            string = clean_str(string)
            words += " ".join(string.split())
        test_df += [words]

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(df)
    sequences = tokenizer.texts_to_sequences(df)

    test_tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    test_tokenizer.fit_on_texts(test_df)
    test_sequences = test_tokenizer.texts_to_sequences(test_df)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    tdata = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    x_test = tdata
    y_test = tlabels

    print('Training and validation set sizes')
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

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    lstm_layer = Bidirectional(LSTM(100))(embedded_sequences)
    predictions = Dense(2, activation='softmax')(lstm_layer)
    model = Model(sequence_input, predictions)
    checkpoint = ModelCheckpoint("weights-biLSTMfinal-{epoch:02d}.hdf5")
    callbacks_list = [checkpoint]
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    print("Model Fittiing - Bidirectional LSTM")
    model.summary()
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=50, callbacks=callbacks_list)

    # Predictions using best weights on testing data
    model.load_weights('weights-biLSTMfinal-04.hdf5')
    predictions = model.predict(x_test, batch_size= 50, verbose =1)
    print("Accuracy score ", accuracy_score(y_test, np.asarray(predictions).round()))

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

count =0
full_count =0
train_val_data = []
test_data = []
with jsonlines.open('instances.jsonl') as reader:
    for obj in reader.iter(type=dict, skip_invalid=True):
        count += 1
        full_count+=1
        if (count > 17600):
            test_data.append(obj)
        if(count<=17600):
            train_val_data.append(obj)

count = 0
truth_data = []
with jsonlines.open('truth.jsonl') as reader:
    for obj in reader.iter(type=dict, skip_invalid=True):
        truth_data.append(obj)

run(train_val_data, test_data, truth_data)