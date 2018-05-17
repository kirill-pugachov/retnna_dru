# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 17:57:31 2018

@author: Kirill
"""


import csv
import math
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing



data_path = ['C:/Users/Кирилл/retnna/Work_data/notreading/', 'C:/Users/Кирилл/retnna/Work_data/reading/']


def read_files(data_path):
    file_number = 0
    female_file_to_work = list()
    male_file_to_work = list()
    for path in data_path:
        for filename in os.listdir(path):
            if filename.startswith("male_"):
                file_number += 1
                male_file_to_work.append(path + filename)
            elif filename.startswith("female_"):
                file_number += 1
                female_file_to_work.append(path + filename)
            else:
                continue
    return (female_file_to_work, male_file_to_work, file_number)


def read_data_from_file(url_file_name):
    with open(url_file_name, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)
        for line in reader:
            yield line


def temp_trans(temp_tank):
    from copy import deepcopy
    temp = deepcopy(temp_tank)
    return temp


def separator(url_file_name):
    result_list = list()
    generator = read_data_from_file(url_file_name)
    line = next(generator)
    while line:
        if float(line[1]) != 0.0 and float(line[2]) != 0.0:
            if float(line[7]) != 0.0 and float(line[8]) != 0.0:
                result_list.append(line)
        try:
            line = next(generator)
        except StopIteration:
            line = None
    return result_list


def all_data_reading(file_name_list):
    result_list = list()
    for url_file_name in file_name_list:
        result_list += separator(url_file_name)
    return result_list


def distance(x, y, x0, y0):
    dist = math.sqrt((x-x0)**2 + (y-y0)**2)
    return dist
    

def one_line_dist(session_list):
    temp = 0
    line = session_list
    if float(line[9]) != 0.0 and float(line[6]) != 0.0:
        temp += distance(
                    float(line[1]), 
                    float(line[2]), 
                    float(line[7]), 
                    float(line[8]))
    return temp


def one_line_speed(session_list):
    dist = 0
    time = 0
    line = session_list
    if float(line[9]) != 0.0 and float(line[6]) != 0.0:
        dist = distance(
                float(line[1]), 
                float(line[2]), 
                float(line[7]), 
                float(line[8]))
        time = float(line[4])
    try:
        dist/time
    except ZeroDivisionError:
        return 0
    else:
        return dist/time
    

def one_line_average_length(session_list):
    line = session_list
    dist = 0
    if float(line[9]) != 0.0 and float(line[6]) != 0.0:
        dist = distance(
                float(line[1]), 
                float(line[2]), 
                float(line[7]), 
                float(line[8]))
    return dist


def one_line_std_sinus(session_list):
    line = session_list
    sin = 0
    if float(line[9]) != 0.0 and float(line[6]) != 0.0:
        hypot = distance(
                float(line[1]), 
                float(line[2]), 
                float(line[7]), 
                float(line[8]))
        kat = float(line[2]) - float(line[8])
        sin = kat/hypot
    return sin
    

def one_line_normal_average_length(session_list):
    temp = 0
    line = session_list
    if float(line[9]) != 0.0 and float(line[6]) != 0.0:
        temp = float(line[1]) - float(line[7])
    return temp   


def one_line_normal_vertical_average_length(session_list):
    temp = 0
    line = session_list
    if float(line[9]) != 0.0 and float(line[6]) != 0.0:
        temp = float(line[2]) - float(line[8])
    return temp


def nominal_session_time(session_list):
    return float(session_list[4])

        
def session_collapsing(session_list):
    horizontal_distance = one_line_dist(session_list)
    average_horizontal_distance = one_line_average_length(session_list)
    inline_speed = one_line_speed(session_list)
    inline_normal_horizontal_distance = one_line_normal_average_length(session_list)
    inline_normal_vertical_distance = one_line_normal_vertical_average_length(session_list)
    session_time = nominal_session_time(session_list)
    one_line_sinus = one_line_std_sinus(session_list)
    return [horizontal_distance, 
            average_horizontal_distance,
            inline_speed,
            inline_normal_horizontal_distance,
            inline_normal_vertical_distance,
            session_time,
            one_line_sinus]


def result_data_cooking(res, param):
    res_list = list()
    for session in res:
        res_list.append([param] + session_collapsing(session))
    return res_list
    

def main(data_path):
    female_file_to_work, male_file_to_work, file_number = read_files(data_path)
    data_list = [female_file_to_work, male_file_to_work]
    for data in data_list:
        if data_list.index(data) == 0:
            param = 0
            female_list = result_data_cooking(all_data_reading(data), param)
        elif data_list.index(data) == 1:
            param = 1
            male_list = result_data_cooking(all_data_reading(data), param)
    return female_list + male_list


def final_list_clean(final_list):
    result = list()
    for line in final_list:
        if float(line[1]) != 0.0:
            if float(line[2]) != 0.0:
                if float(line[5]) != 0.0:
                    if float(line[6]) != 0.0:
                        result.append(line)
    return result
                

def list_to_frame(final_list):
    df = pd.DataFrame(final_list, columns=['Target',
                                           'X0', 
                                           'X1', 
                                           'X2', 
                                           'X3', 
                                           'X4', 
                                           'X5', 
                                           'X6'])
    return df


def short_soft_voting(df_res, y):
    clf2 = ensemble.RandomForestClassifier(n_estimators=200,
                                           max_depth=5,
                                           max_features=1, 
                                           random_state=42, 
                                           n_jobs=-1)
    clf3 = ensemble.GradientBoostingClassifier(n_estimators=200, 
                                              learning_rate=0.01, 
                                              max_depth=5, 
                                              random_state=42)
    clf1 = GaussianNB(priors=None)
    clf4 = SGDClassifier(max_iter=500,
                         alpha=0.1,
                         tol=1e-4, 
                         shuffle=True, 
                         penalty='l2', 
                         loss='log')
    clf5 = DecisionTreeClassifier(max_depth=7)
    clf8 = LogisticRegression(C=1e5)
    clf9 = MLPClassifier(
            hidden_layer_sizes=(40, 40, 40, 40),
            learning_rate='constant',
            alpha=0.0001,
            max_iter=600,
            activation='logistic')
    clf10 = ensemble.AdaBoostClassifier()
    clf11 = KNeighborsClassifier(3)
    eclf = VotingClassifier(estimators=[('gau', clf1), ('rfc', clf2), 
                                        ('gbs', clf3), ('sgdc', clf4),
                                        ('dtc', clf5), ('LogReg', clf8),
                                        ('Neuro_mlp', clf9), ('AdaBoost', clf10),
                                        ('KNeighbors', clf11)], 
                                voting='soft', weights=[1,1,1,1,1,1,1,1,1])
    for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf8, clf9, clf10, clf11, eclf], 
                          ['GaussianNB', 
                           'RandomForestClassifier', 
                           'GradientBoosting', 
                           'SGDClassifier', 
                           'DecisionTreeClassifier',
                           'LogReg',
                           'Neuro_MLP',
                           'AdaBoost',
                           'KNeighbors',
                           'Ensemble']):
        scores = cross_val_score(clf, df_res, y, cv=5, n_jobs=-1, scoring='roc_auc')
        print("ROC_AUC scoring: %0.3f (+/- %0.3f) [%s]" % (scores.mean(), scores.std(), label))
        

def soft_voting(df_res, y):
    clf2 = ensemble.RandomForestClassifier(n_estimators=300,
                                           max_depth=5,
                                           max_features=1, 
                                           random_state=42, 
                                           n_jobs=-1)
    clf3 = ensemble.GradientBoostingClassifier(n_estimators=300, 
                                              learning_rate=0.01, 
                                              max_depth=5, 
                                              random_state=42)
    clf1 = GaussianNB(priors=None)
    clf4 = SGDClassifier(max_iter=1000,
                         alpha=0.01,
                         tol=1e-4, 
                         shuffle=True, 
                         penalty='l2', 
                         loss='log')
    clf5 = DecisionTreeClassifier(max_depth=7)
    clf6 = SVC(kernel="linear", C=0.025, probability=True)
    clf7 = SVC(gamma=2, C=1, probability=True)
    clf8 = LogisticRegression(C=1e5)
    clf9 = MLPClassifier(hidden_layer_sizes=(35, 30, 25, 20, 15, 10, 5), alpha=0.0001, max_iter=5000, activation='logistic')
    clf10 = ensemble.AdaBoostClassifier()
    clf11 = KNeighborsClassifier(3)
    clf12 = QuadraticDiscriminantAnalysis() #GaussianProcessClassifier(1.0 * RBF(1.0))
    eclf = VotingClassifier(estimators=[('gau', clf1), ('rfc', clf2), 
                                        ('gbs', clf3), ('sgdc', clf4),
                                        ('dtc', clf5), ('svm_linear', clf6),
                                        ('svm_gamma', clf7), ('LogReg', clf8),
                                        ('Neuro_mlp', clf9), ('AdaBoost', clf10),
                                        ('KNeighbors', clf11), ('QDA', clf12)], 
                                voting='soft', weights=[1,1.2,1.2,1,1.2,1,1,1.2,1.2,1,1,1])
    for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, clf10, clf11, clf12, eclf], 
                          ['GaussianNB', 
                           'RandomForestClassifier', 
                           'GradientBoosting', 
                           'SGDClassifier', 
                           'DecisionTreeClassifier',
                           'SVC_linear',
                           'SVC_gamma',
                           'LogReg',
                           'Neuro_MLP',
                           'AdaBoost',
                           'KNeighbors',
                           'QuadraticDA',
                           'Ensemble']):
        scores = cross_val_score(clf, df_res, y, cv=5, n_jobs=-1, scoring='roc_auc')
        print("ROC_AUC scoring: %0.3f (+/- %0.3f) [%s]" % (scores.mean(), scores.std(), label))


def data_preprocessing(X):
    poly = preprocessing.PolynomialFeatures(interaction_only=True)
    X_proc = poly.fit_transform(X)
    standart_scaler = preprocessing.StandardScaler()
    return standart_scaler.fit_transform(X_proc)


if __name__ == '__main__':

    final_list = main(data_path)
    total_dataframe = list_to_frame(final_list)
    y = total_dataframe['Target']
    X = total_dataframe.drop('Target', 1)
    X_proc = data_preprocessing(X)    
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, 
                                                        test_size=0.3, 
                                                        random_state=42, 
                                                        shuffle=True)
    short_soft_voting(X_train, y_train)