#!/usr/bin/env python

import jsonlines
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import linear_model
from tqdm import tqdm


def run(train_data, valid_data, test_data, truth_data):
    train_data_df = pd.DataFrame.from_dict(train_data)
    truth_data_df = pd.DataFrame.from_dict(truth_data)
    train = pd.merge(train_data_df, truth_data_df, on="id")
    data = train.values
    print("Training data length", len(data))

    df = []
    y = []

    for i in data:
        if(i[9]=="clickbait"):
            y.append(1)
        else:
            y.append(0)

    textFeatures = ["postText", "targetCaptions", "targetParagraphs", "targetTitle", "targetKeywords", "targetDescription"]
    textColumns = train[textFeatures]
    textColumns = textColumns.values.tolist()

    for i in range(len(textColumns)):
        text = []
        for j in range(0,5):
            k = textColumns[i][j]
            if (j == 3 or j == 4):
                text.append(k)
            else:
                text+=k
        words = ""
        for string in text:
            string = clean_str(string)
            words +=" ".join(string.split())
        df+=[words]

    vectorizer = CountVectorizer(input='content', lowercase=False, analyzer='word', stop_words='english')
    X = vectorizer.fit_transform(df)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X)
    print(X_train_tfidf.shape)

    model = linear_model.LinearRegression()
    model.fit(X_train_tfidf, y)

    ### VALIDATION DATA ###

    print("Validation")
    valid_data_df = pd.DataFrame.from_dict(valid_data)
    valid = pd.merge(valid_data_df, truth_data_df, on="id")
    vdata = valid.append(train).values

    y_valid = []
    for i in vdata:
        if (i[9] == "clickbait"):
            y_valid.append(1)
        if (i[9] == "no-clickbait"):
            y_valid.append(0)

    y_valid = pd.DataFrame(y_valid)
    print("Y_valid length", len(y_valid))
    vdata = valid[textFeatures].append(train[textFeatures]).values.tolist()

    df_valid = []
    for i in range(len(vdata)):
        text = []
        for j in range(0, 5):
            k = vdata[i][j]
            if (j == 3 or j == 4):
                text.append(k)
            else:
                text += k
        words = ""
        for string in text:
            string = clean_str(string)
            words += " ".join(string.split())
        df_valid += [words]

    # a_train, a_val, b_train, b_val = train_test_split(df_valid, y_valid, test_size = 0.11, random_state = 42)
    predicted = []
    for v in df_valid:
        valid_X = vectorizer.transform([v])
        X_valid_tfidf = tfidf_transformer.transform(valid_X)
        predicted.append(model.predict(X_valid_tfidf).round())

    scores = accuracy_score(y_valid, predicted)
    print("Validation Data Accuracy ", scores)

    ### TEST DATA ###

    test_data_df = pd.DataFrame.from_dict(test_data)
    test = pd.merge(test_data_df, truth_data_df, on="id")
    tdata = test.values
    print("length of test data")
    print(len(tdata))

    y_test =[]
    df_test =[]

    for i in tdata:
        if(i[9]=="clickbait"):
            y_test.append(1)
        if(i[9]=="no-clickbait"):
            y_test.append(0)

    textColumns_test = test[textFeatures]
    textColumns_test = textColumns_test.values.tolist()

    for i in range(len(textColumns_test)):
        text = []
        for j in range(0,5):
            k = textColumns_test[i][j]
            if (j == 3 or j == 4):
                text.append(k)
            else:
                text+=k
        words = ""
        for string in text:
            string = clean_str(string)
            words +=" ".join(string.split())
        df_test+=[words]

    predicted = []
    for t in df_test:
        test_X = vectorizer.transform([t])
        X_test_tfidf = tfidf_transformer.transform(test_X)
        predicted.append(model.predict(X_test_tfidf).round())

    scores = accuracy_score(y_test, predicted)
    print("Test Data Accuracy ", scores)

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

count = 0
train_data = []
valid_data = []
test_data = []
with jsonlines.open('instances.jsonl') as reader:
    for obj in reader.iter(type=dict, skip_invalid=True):
        count += 1
        if (count > 15630 and count <= 17584):
            valid_data.append(obj)
        if (count > 17584):
            test_data.append(obj)
        if(count<=15630):
            train_data.append(obj)

count = 0
truth_data = []
with jsonlines.open('truth.jsonl') as reader:
    for obj in reader.iter(type=dict, skip_invalid=True):
        truth_data.append(obj)

run(train_data, valid_data, test_data, truth_data)