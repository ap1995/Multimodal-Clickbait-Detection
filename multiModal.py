import numpy as np
import pandas as pd
import re
import jsonlines
import json

import sys
import os

os.environ['KERAS_BACKEND'] = 'theano'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers.merge import concatenate
from keras.layers import Add

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
#from attention_decoder import AttentionDecoder
import AttentionwithContext as ac
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
# from keras import initializations

import tensorflow as tf
from config import Config
from cnn_model import cnn_model

def run(train_data, test_data, truth_data):
    final_vals = []
    data_df = pd.DataFrame.from_dict(train_data)
    truth_data_df = pd.DataFrame.from_dict(truth_data)
    train = pd.merge(data_df, truth_data_df, on="id")
    features = ["id", "postMedia", "targetTitle", "truthClass"]
    vals = train[features]
    vals = vals.values.tolist()
    for i in range(len(vals)):
        if vals[i][1] != []:
            final_vals.append([vals[i][0], [vals[i][1][0]], [vals[i][2]], vals[i][3]])

    vals_df = pd.DataFrame(final_vals, columns=["id", "file_path", "title", "truthClass"])

    image_features = imgModel(vals_df)
    print("-------------------xtrainshrpe")
    print(type(image_features[0][0]))
    finalTestvals = []
    test_data_df = pd.DataFrame.from_dict(test_data)
    test = pd.merge(test_data_df, truth_data_df, on="id")
    test_vals = test[features].values.tolist()
    for i in range(len(test_vals)):
        if test_vals[i][1] != []:
            finalTestvals.append([test_vals[i][0], [test_vals[i][1][0]], [test_vals[i][2]], test_vals[i][3]])
    tdata = test[features].values
    tdata = test_data_df.values

    labels = []
    tlabels = []
    df = []

    for i in vals_df.values:
        if(i[3]=="clickbait"):
            labels.append(1)
        else:
            labels.append(0)

    for i in tdata:
        if (i[3] == "clickbait"):
            tlabels.append(1)
        else:
            tlabels.append(0)

    for i in range(vals_df.values.shape[0]):
        text = []
        for j in range(1,5):
            k = vals_df.values[i][2]
            text+=(k)
        words = ""
        for string in text:
            string = clean_str(string)
            words +=" ".join(string.split())
        df+=[words]

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(df)
    sequences = tokenizer.texts_to_sequences(df)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

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


    image_features_train = image_features[:-nb_validation_samples]
    image_features_val = image_features[-nb_validation_samples:]

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

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')

    image_data_input = Input(shape=(100352,),dtype='float32')
    embedded_sequences = embedding_layer(sequence_input)
    l_lstm = Bidirectional(LSTM(100))(embedded_sequences)

    print("-----------------------------")
    print(l_lstm.shape)

    preds_text = Dense(2, activation='softmax')(l_lstm)

    preds_image = Dense(2,activation='softmax')(image_data_input)

    preds_add = Add()([preds_text,preds_image])

    preds = Dense(2)(preds_add)


    #preds = preds_text

    model1 = Model([sequence_input,image_data_input], preds)
    model1.add_update(ac.AttentionWithContext()) ###############

    checkpoint = ModelCheckpoint("weights-bilstm-{epoch:02d}-{val_acc:.2f}.hdf5")
    callbacks_list = [checkpoint]
    model1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    print("model fitting - Bidirectional LSTM with Attention")
    model1.summary()
    print('------')

    model1.fit([x_train, np.array(image_features_train)], y_train, validation_data=([x_val, np.array(image_features_val)], y_val), epochs=10, batch_size=50, callbacks=callbacks_list)

def imgModel(vals_df):
    # model = VGG16(weights='imagenet', include_top=False)
    # model.summary()
    img_features = []
    config = Config()
    images = tf.placeholder(
        dtype=tf.float32,
        shape=[config.batch_size] + config.image_shape)

    sess = tf.Session()

    model = cnn_model(config)
    features = model.build_vgg16(images)
    model.load_cnn(sess,config.vgg16_file)

    for entry in vals_df.values:
        img_path = entry[1][0]
        # print(img_path)
        img = image.load_img(img_path, target_size=(224, 224,3))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        imgList = img_data

        # vgg16_feature = model.predict(img_data)

        vgg16_feature = sess.run(features,feed_dict={images:img_data})
        # print("----------------------------------------------------------")
        # print(type(vgg16_feature))
        img_features.append(vgg16_feature[0])

    return img_features

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
LIMIT = 100
train_val_data = []
test_data = []

with jsonlines.open('instances.jsonl') as reader:
    for obj in reader.iter(type=dict, skip_invalid=True):
        count += 1
        if (count > 90 and count<=LIMIT): #9275
            test_data.append(obj)
        if(count<=90):
            train_val_data.append(obj)

count = 0
truth_data = []
with jsonlines.open('truth.jsonl') as reader:
    for obj in reader.iter(type=dict, skip_invalid=True):
        truth_data.append(obj)

run(train_val_data, test_data, truth_data)
