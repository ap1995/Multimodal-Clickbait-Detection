#!/usr/bin/env python

import jsonlines
from pprint import pprint
import unicodedata
import csv
import numpy as np
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn import linear_model

def run(train_data, valid_data, test_data, train_truth, valid_truth, test_truth):
    # data = list(train_data)
    train_data_df = pd.DataFrame.from_dict(train_data)
    train_truth_df = pd.DataFrame.from_dict(train_truth)
    train = pd.merge(train_data_df, train_truth_df, on="id")
    # print(type(train["targetDescription"][0][0]))
    # train = pd.DataFrame[{"x":train_data_df, "y":train_truth_df[["truthMean"]]}]
    # train.x = train.x.apply(lambda x: list(map(int, x)))
    # y=[]
    # for i in data:
    #     y.append(i[0])

    # print(df[0])
    # df =[]

    textColumns = train[["postText", "targetCaptions", "targetParagraphs", "targetTitle", "postTimestamp","targetKeywords", "targetDescription"]]
    textColumns = textColumns.values.T.tolist()
    # mapper = DataFrameMapper([(textColumns.columns, StandardScaler())])
    # scaled_features = mapper.fit_transform(textColumns.copy())
    # print(textColumns.shape)
    vectorizer = CountVectorizer(input='content', lowercase=False, analyzer='word', stop_words='english')
    # X =vectorizer.fit_transform(textColumns.values)
    X = vectorizer.fit_transform(textColumns)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X)
    # print(X_train_tfidf.shape)
    # print(train["truthClass"].shape)

    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    clf = LinearSVC() #random_state=0
    clf.fit(X_train_tfidf, train[["truthMean"]])

    ### VALIDATION DATA ###
    #
    # valid_data = list(valid_data)
    #
    # valid =[]
    # valid_y = []
    # for i in valid_data:
    #     valid_y.append(i[0])
    # for i in valid_data:
    #     valid += [' '.join(i)]
    #
    # valid_X = vectorizer.fit_transform(valid)
    # X_valid_tfidf = tfidf_transformer.fit_transform(valid_X)
    #
    # print(X_valid_tfidf.shape)
    # print(clf.score(X_valid_tfidf, valid_y))

    #### TEST DATA ###

    test_data = list(test_data)

    test =[]
    for i in test_data:
        test += [' '.join(i)]

    text_file = open(output_file, "w")
    for t in test:
        test_X = vectorizer.transform([t])
        X_test_tfidf = tfidf_transformer.transform(test_X)
        predicted = clf.predict(X_test_tfidf)
        # text_file.write(predicted[0]+'\n')
        print(accuracy_score(test_truth["truthMean"], predicted))
# with open('instances.jsonl', encoding='utf-8') as data_file:
#     data = json.loads(data_file.read())
# json_file = open('instances.jsonl', encoding='utf-8')
# text = json_file.read()
# jdata = json.load(json_file)

# for postMedia, postText, id, targetCaptions, targetParagraphs, targetTitle, postTimestamp, targetKeywords, targetDescription in jdata.items():
#    pprint("Key:")
#    pprint(id)
count =0
train = []
valid = []
test = []
with jsonlines.open('instances.jsonl') as reader:
    for obj in reader.iter(type=dict, skip_invalid=True):
        train.append(obj)
        count+=1
        if(count >=15630 and count <=17584):
            valid.append(obj)
        if(count>17584):
            test.append(obj)
# print(len(train))
# print(len(valid))
# print(len(test))
count=0
train_truth = []
valid_truth = []
test_truth = []
with jsonlines.open('truth.jsonl') as reader:
    for obj in reader.iter(type=dict, skip_invalid=True):
        train_truth.append(obj)
        count += 1
        if (count >= 15630 and count <= 17584):
            valid_truth.append(obj)
        if (count > 17584):
            test_truth.append(obj)
# fp = io.BytesIO()  # file-like object
# with jsonlines.Writer(fp) as writer:
#     writer.write(...)
# fp.close()

train_data = train
valid_data = valid
test_data = test
output_file = 'predictions.txt'
run(train_data, valid_data, test_data, train_truth, valid_truth, test_truth)