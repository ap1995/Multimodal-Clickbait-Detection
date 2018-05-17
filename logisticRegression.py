#!/usr/bin/env python

import jsonlines
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import linear_model
from tqdm import tqdm


def run(train_data, valid_data, test_data, truth_data):
    train_data_df = pd.DataFrame.from_dict(train_data)
    truth_data_df = pd.DataFrame.from_dict(truth_data)
    train = pd.merge(train_data_df, truth_data_df, on="id")
    data = train.values

    textFeatures = ["postText", "targetCaptions", "targetParagraphs", "targetTitle", "targetKeywords",
                    "targetDescription", "truthClass"]

    vals = data.tolist()
    final_vals = []
    # print(vals[0])
    for i in range(len(vals)):
        if vals[i][1] != []:
            print(vals[i][2])
            final_vals.append([vals[i][2], vals[i][4], vals[i][5], vals[i][6], vals[i][7], vals[i][8], vals[i][9]])

    vals_df = pd.DataFrame(final_vals, columns=["postText", "targetCaptions", "targetParagraphs", "targetTitle", "targetKeywords",
                    "targetDescription", "truthClass"])
    textColumns = vals_df.values.tolist()

    df = []
    y = []
    print('---------')
    print(len(final_vals))

    VALIDATION_SPLIT = 0.1
    nb_validation_samples = int(VALIDATION_SPLIT * len(final_vals))
    valid_data = final_vals[:nb_validation_samples]
    test_data = final_vals[int(0.8 * len(final_vals)):int(0.9 * len(final_vals))]
    final_vals = final_vals[0:int(len(final_vals)*0.8)]

    for i in final_vals:
        if(i[6]=="clickbait"):
            y.append(1)
        else:
            y.append(0)
    # print(textColumns[0])

    for i in range(len(final_vals)):
        text = []
        for j in range(0,6):
            k = final_vals[i][j]
            # print(k, j)
            if (j == 2 or j == 3):
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

    clf = linear_model.LinearRegression()
    clf.fit(X_train_tfidf, y)

    ### VALIDATION DATA ###

    print("Validation")
    # valid_data_df = pd.DataFrame(valid_data)
    # valid_data_df = pd.DataFrame.from_dict(valid_data)
    # valid = pd.merge(valid_data_df, truth_data_df, on="id")
    # vdata = valid.append(train).values
    # vdata = final_vals.append(valid_data_df.values).tolist()
    vdata = final_vals + valid_data

    y_valid = []
    for i in vdata:
        if (i[6] == "clickbait"):
            y_valid.append(1)
        if (i[6] == "no-clickbait"):
            y_valid.append(0)

    y_valid = pd.DataFrame(y_valid)
    print("Y_valid length", len(y_valid))
    # vdata = valid[textFeatures].append(train[textFeatures]).values.tolist()

    df_valid = []
    for i in range(len(vdata)):
        text = []
        for j in range(0, 5):
            k = vdata[i][j]
            if (j == 2 or j == 3):
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
        predicted.append(clf.predict(X_valid_tfidf).round())

    scores = accuracy_score(y_valid, predicted)
    print("Validation Data Accuracy ", scores)

    ### TEST DATA ###

    # predicted = []
    # for t in df_test:
    #     test_X = vectorizer.transform([t])
    #     X_test_tfidf = tfidf_transformer.transform(test_X)
    #     predicted.append(model.predict(X_test_tfidf).round())
    #
    # scores = accuracy_score(y_test, predicted)

    tdata = test_data

    y_test =[]
    df_test =[]

    for i in tdata:
        if(i[6]=="clickbait"):
            y_test.append(1)
        if(i[6]=="no-clickbait"):
            y_test.append(0)

    # textColumns_test = test[textFeatures]
    # textColumns_test = textColumns_test.values.tolist()

    for i in range(len(tdata)):
        text = []
        for j in range(0,5):
            k = tdata[i][j]
            if (j == 2 or j == 3):
                text.append(k)
            else:
                text+=k
        words = ""
        for string in text:
            string = clean_str(string)
            words +=" ".join(string.split())
        df_test+=[words]

    # test_X = vectorizer.fit_transform(df_test)
    # X_test_tfidf = tfidf_transformer.fit_transform(test_X)
    # predicted = model.predict(X_test_tfidf)
    # print(clf.score(X_test_tfidf, y_test))
    predicted = []
    for t in df_test:
        test_X = vectorizer.transform([t])
        X_test_tfidf = tfidf_transformer.transform(test_X)
        predicted.append(clf.predict(X_test_tfidf).round())

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
        train_data.append(obj)
        # if (count > 15630 and count <= 17584):
        #     valid_data.append(obj)
        # if (count > 17584):
        #     test_data.append(obj)
        # if(count<=15630):
        #     train_data.append(obj)

count = 0
truth_data = []
with jsonlines.open('truth.jsonl') as reader:
    for obj in reader.iter(type=dict, skip_invalid=True):
        truth_data.append(obj)

run(train_data, valid_data, test_data, truth_data)