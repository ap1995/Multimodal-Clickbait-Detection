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
from sklearn.metrics import accuracy_score

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import Embedding, Merge, Dropout, LSTM, Bidirectional
from keras.models import Model
import AttentionwithContext as ac

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
    print("Final vals length", len(final_vals))

    finalTestvals = []
    test_data_df = pd.DataFrame.from_dict(test_data)
    test = pd.merge(test_data_df, truth_data_df, on="id")
    test_vals = test[features].values.tolist()
    for i in range(len(test_vals)):
        if test_vals[i][1] != []:
            finalTestvals.append([test_vals[i][0], [test_vals[i][1][0]], [test_vals[i][2]], test_vals[i][3]])

    test_vals_df = pd.DataFrame(finalTestvals, columns=["id", "file_path", "title", "truthClass"])
    print("finalTestVals length", len(finalTestvals))

    labels = []
    tlabels = []
    df = []

    image_features = imgModel(vals_df)

    for i in vals_df.values:
        if(i[3]=="clickbait"):
            labels.append(1)
        else:
            labels.append(0)

    for i in test_vals_df.values:
        if (i[3] == "clickbait"):
            tlabels.append(1)
        else:
            tlabels.append(0)

    for i in range(vals_df.values.shape[0]):
        text = []
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

    t_df = [] #### Test data
    for i in range(test_vals_df.values.shape[0]):
        text = []
    # for j in range(1,5):
        k = test_vals_df.values[i][2]
        text+=(k)
        words = ""
        for string in text:
            string = clean_str(string)
            words +=" ".join(string.split())
        t_df+=[words]

    tokenizer_test = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer_test.fit_on_texts(t_df)
    test_sequences = tokenizer_test.texts_to_sequences(t_df)

    word_index = tokenizer.word_index  ## For training data
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

    image_features_train = image_features[:-nb_validation_samples]
    image_features_val = image_features[-nb_validation_samples:]
    image_features_test = imgModel(test_vals_df)

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

    preds_image = Dense(2, activation='softmax')(image_data_input)

    # preds_add = Add()([preds_text,preds_image])
    preds_add = concatenate([preds_text, preds_image], axis=-1)

    preds = Dense(2)(preds_add)

    model = Model([sequence_input,image_data_input], preds)

    checkpoint = ModelCheckpoint("weights-multimodal-{epoch:02d}-{val_acc:.2f}.hdf5")
    callbacks_list = [checkpoint]
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    print("model fitting - Bidirectional LSTM (Multimodalbaseline)")
    model.summary()
    print('------')

    model.fit([x_train, np.array(image_features_train)], y_train, validation_data=([x_val, np.array(image_features_val)], y_val), epochs=5, batch_size=50, callbacks=callbacks_list)

    model.load_weights('weights-multimodal-05-0.81.hdf5')
    preds = model.predict([x_test, np.array(image_features_test)], batch_size=50, verbose=1)
    preds_new = []
    for i in range(len(preds)):
        preds_new.append(preds[i][0] + preds[i][1])
    print("Accuracy score on Test data ", accuracy_score(y_test, np.asarray(preds_new).round()))

def imgModel(vals_df):
    # model = VGG16(weights='imagenet', include_top=False)
    # model.summary()
    img_features = []
    config = Config()
    images = tf.placeholder(
        dtype=tf.float32,
        shape=[config.batch_size] + config.image_shape)
    # tf.reset_default_graph()
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

        # vgg16_feature = model.predict(img_data)

        vgg16_feature = sess.run(features,feed_dict={images:img_data})

        img_features.append(vgg16_feature[0])

    tf.reset_default_graph()
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
full_count = 0
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
