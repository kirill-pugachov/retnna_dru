# -*- coding: utf-8 -*-
"""
Created on Thu May  3 10:48:29 2018

@author: Kirill
"""


import csv
import math
import os

import pandas as pd
#import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#from sklearn import ensemble
#from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussianNB
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.svm import SVC
#from sklearn.neural_network import MLPClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


url_data = 'C:/Users/Кирилл/retnna/Data/csv_data/notreading/'
file_name = 'pct2_olga.csv' #'faces_anastasia.csv'
data_path = ['C:/Users/Кирилл/retnna/Work_data/notreading/', 'C:/Users/Кирилл/retnna/Work_data/reading/']


###proceed data to NN
def find_coef_straight(x1, y1, x2, y2):
#    print(x1, y1, x2, y2)
    a = y2 - y1
    b = x1 - x2
    c = -x1*(y2-y1)+y1*(x2-x1)
    return (a, b, c)


def y_count(a, b, c, x):
    return round(- (a * x)/b - c/b)


def scalling(x, koef):
    if math.fabs(x) < 1:
        return round(math.fabs(x) * koef)
    else:
        return round(0.999 * koef)


def session_collapsing(session_list):
    pic_size = (80, 60)
    point_color = 1
    res = [0.01 for i in range(pic_size[0] * pic_size[1])]
    for session in session_list:
        x1 = scalling(float(session[1]), pic_size[0])
        x2 = scalling(float(session[7]), pic_size[0])
        y1 = scalling(float(session[2]), pic_size[1])
        y2 = scalling(float(session[8]), pic_size[1])
        if x1 != 0 and y1 != 0:
            if x2 != 0 and y2 != 0:
                a, b, c = find_coef_straight(x1, y1, x2, y2)
                print(1, x1, y1, x2, y2, a, b, c)
                if b < 0:
                    for x in range(x1, x2 + 1):
                        y = y_count(a, b, c, x)
                        res[(y - 1) * pic_size[0] + x - 1] += point_color
                elif b > 0:
                    for x in range(x2, x1 + 1):
                        y = y_count(a, b, c, x)
                        res[(y - 1) * pic_size[0] + x - 1] += point_color
                else:
                    res[(y1-1) * pic_size[0] + x1 - 1] += point_color
                    res[(y2-1) * pic_size[0] + x2 - 1] += point_color
            elif x2 == 0 and y2 != 0:
                res[(y1-1) * pic_size[0] + x1 - 1] += point_color
                res[(y2-1) * pic_size[0] - 1] += point_color
            elif x2 != 0 and y2 == 0:
                res[(y1-1) * pic_size[0] + x1 - 1] += point_color
                res[x2 - 1] += point_color
            elif x2 == 0 and y2 == 0:
                res[(y1-1) * pic_size[0] + x1 - 1] += point_color
        elif x1 == 0 and y1 != 0:
            if x2 != 0 and y2 != 0:
                a, b, c = find_coef_straight(x1, y1, x2, y2)
                print(22, x1, y1, x2, y2, a, b, c)

                res[(y1-1) * pic_size[0] + x1 - 1] += point_color
                res[(y2-1) * pic_size[0] + x2 - 1] += point_color
            elif x2 == 0 and y2 != 0:
                res[(y1-1) * pic_size[0] + x1 - 1] += point_color
                res[(y2-1) * pic_size[0] - 1] += point_color
            elif x2 != 0 and y2 == 0:
                res[(y1-1) * pic_size[0] + x1 - 1] += point_color
                res[x2 - 1] += point_color
            elif x2 == 0 and y2 == 0:
                res[(y1-1) * pic_size[0] + x1 - 1] += point_color
        elif x1 != 0 and y1 == 0:
            if x2 != 0 and y2 != 0:
                a, b, c = find_coef_straight(x1, y1, x2, y2)
                print(333, x1, y1, x2, y2, a, b, c)

                res[(y1-1) * pic_size[0] + x1 - 1] += point_color
                res[(y2-1) * pic_size[0] + x2 - 1] += point_color
            elif x2 == 0 and y2 != 0:
                res[(y1-1) * pic_size[0] + x1 - 1] += point_color
                res[(y2-1) * pic_size[0] - 1] += point_color
            elif x2 != 0 and y2 == 0:
                res[(y1-1) * pic_size[0] + x1 - 1] += point_color
                res[x2 - 1] += point_color
            elif x2 == 0 and y2 == 0:
                res[(y1-1) * pic_size[0] + x1 - 1] += point_color
        elif x1 == 0 and y1 == 0:
            if x2 != 0 and y2 != 0:
                a, b, c = find_coef_straight(x1, y1, x2, y2)
                print(4444, x1, y1, x2, y2, a, b, c)

                res[(y1-1) * pic_size[0] + x1 - 1] += point_color
                res[(y2-1) * pic_size[0] + x2 - 1] += point_color
            elif x2 == 0 and y2 != 0:
                res[(y1-1) * pic_size[0] + x1 - 1] += point_color
                res[(y2-1) * pic_size[0] - 1] += point_color
            elif x2 != 0 and y2 == 0:
                res[(y1-1) * pic_size[0] + x1 - 1] += point_color
                res[x2 - 1] += point_color
            elif x2 == 0 and y2 == 0:
                res[(y1-1) * pic_size[0] + x1 - 1] += point_color
                
                

    return res

def session_collapsing_old(session_list):
    pic_size = (80, 60)
    point_color = 80
    res = [0.01 for i in range(pic_size[0] * pic_size[1])]
    for session in session_list:
        if float(session[1]) != 0.0 and float(session[2]) != 0.0:
            if float(session[7]) != 0.0 and float(session[8]) != 0.0:
                a, b, c = find_coef_straight(
                        round(float(session[1]) * pic_size[0]),
                        round(float(session[2]) * pic_size[1]),
                        round(float(session[7]) * pic_size[0]),
                        round(float(session[8]) * pic_size[1]))
                print(a, b, c)
                for x in range(round(float(session[1]) * pic_size[0]), round(float(session[7]) * pic_size[0]) + 1):                    
                    y = int(y_count(a, b, c, x))
                    print(y, x)
                    if y == 0:
                        res[y * pic_size[0] + x] += point_color
                    elif y < 0:
                        res[int((math.fabs(pic_size[1] + y) - 1) * pic_size[0] - x)] += point_color
                    elif y > 0:
                        res[(y - 1) * pic_size[0] + x] += point_color
            else:
                res[int((round(float(session[2]) * pic_size[1]) - 1) * pic_size[0] + round(float(session[1]) * pic_size[0]))] += point_color
        elif float(session[7]) != 0.0 and float(session[8]) != 0.0:
            res[int((int(float(session[8]) * pic_size[1]) - 1) * pic_size[0] + int(float(session[7]) * pic_size[0]))] += point_color
        else:
            res[0] = 0.01
    return res


###read data from files
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
    temp_result_list =list()
    result_list = list()
    generator = read_data_from_file(url_file_name)
    line = next(generator)
    FPOGDS = line[3]
    FPOGID = line[5]
    FPOGDS_current = FPOGDS
    FPOGID_current = FPOGID
    while line:
        if FPOGDS_current == FPOGDS and FPOGID_current == FPOGID:
            temp_result_list.append(line)
            FPOGDS = line[3]
            FPOGID = line[5]
        else:
            result_list.append(temp_trans(temp_result_list))
            temp_result_list.clear()
            FPOGDS = line[3]
            FPOGID = line[5]
            FPOGDS_current = FPOGDS
            FPOGID_current = FPOGID
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


def list_to_frame(final_list):
    df = pd.DataFrame(final_list)
    return df


def data_ready(data_path):
    final_list = main(data_path)
    total_dataframe = list_to_frame(final_list)
    y = total_dataframe['Target']
    X = total_dataframe.drop('Target', 1)
    standart_scaler = preprocessing.StandardScaler() #0.629
    X_proc = standart_scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, 
                                                        test_size=0.3, 
                                                        random_state=42, 
                                                        shuffle=True)
    return X_train, X_test, y_train, y_test


def select_params_for_SGDClassifier(df_res, y):
    alpha_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100]
    MAX_iter = [100, 1000, 2000, 5000, 10000]
    loss_list = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
    for alph in alpha_list:
        for it_r in MAX_iter:
            for loss_cr in loss_list:
                label = 'SGDClassifier'
                clf = SGDClassifier(alpha=alph, max_iter=it_r, loss=loss_cr)
                scores = cross_val_score(clf, df_res, y, cv=10, n_jobs=-1, scoring='roc_auc')
                print("ROC_AUC scoring: %0.3f (+/- %0.3f) [%s] [%s] [%s] [%s]" % (scores.mean(), 
                                               scores.std(), 
                                               label, 
                                               alph, 
                                               it_r, 
                                               loss_cr))    


def select_params_for_LogisticRegression(df_res, y):
    CN_list = [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
    MAX_iter = [10, 100, 1000]
    solvers = ['newton-cg', 'lbfgs', 'sag']
    for CN in CN_list:
        for it_r in MAX_iter:
            for solve in solvers:
                label = 'LogReg'
                clf = LogisticRegression(C=CN, max_iter=it_r, solver=solve)
                scores = cross_val_score(clf, df_res, y, cv=10, n_jobs=-1, scoring='roc_auc')
                print("ROC_AUC scoring: %0.3f (+/- %0.3f) [%s] [%s] [%s] [%s]" % (scores.mean(), 
                                               scores.std(), 
                                               label, 
                                               CN, 
                                               it_r, 
                                               solve))
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = data_ready(data_path)
    print('\n', 'SGDClassifier', '\n')
    select_params_for_SGDClassifier(X_train, y_train)
    print('\n', 'LogisticRegression', '\n')
    select_params_for_LogisticRegression(X_train, y_train)
 

#                print(session)
#                print(float(session[1]), float(session[2]), float(session[7]), float(session[8]))
#                print(x1, y1, x2, y2, a, b, c)
#                print((y1-1) * pic_size[0] + x1)
#                print((y2-1) * pic_size[0] + x2)    