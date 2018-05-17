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



#url_data = 'C:/Users/Кирилл/retnna/Data/csv_data/raw_nr/'
#file_name = 'Search_Andrii2_all_gaze.csv'

url_data = 'C:/Users/Кирилл/retnna/Data/csv_data/notreading/'
file_name = 'pct2_olga.csv' #'faces_anastasia.csv'
data_path = ['C:/Users/Кирилл/retnna/Work_data/notreading/', 'C:/Users/Кирилл/retnna/Work_data/reading/']

#data = url_data + file_name


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


def read_data(url_data, file_name):
    f_open = open(url_data + file_name)
    return pd.read_csv(f_open, sep='\t', encoding='cp1251')


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


def distance(x, y, x0, y0):
    dist = math.sqrt((x-x0)**2 + (y-y0)**2)
    return dist
    

def one_line_dist(session_list):
    temp = 0
    for line in session_list:
        if float(line[9]) != 0.0 and float(line[6]) != 0.0:
            temp += distance(
                    float(line[1]), 
                    float(line[2]), 
                    float(line[7]), 
                    float(line[8]))
        else:
            break
    return temp


def one_line_speed(session_list):
    temp = list()
    for line in session_list:
        if float(line[9]) != 0.0 and float(line[6]) != 0.0:
            dist = distance(
                    float(line[1]), 
                    float(line[2]), 
                    float(line[7]), 
                    float(line[8]))
            time = float(line[4])
            temp.append(dist/time)
        else:
            break
    try:
        sum(temp)/len(temp)
    except ZeroDivisionError:
        return 0
    else:
        return sum(temp)/len(temp)


def one_line_std_speed(session_list):
    temp = list()
    for line in session_list:
        if float(line[9]) != 0.0 and float(line[6]) != 0.0:
            dist = distance(
                    float(line[1]), 
                    float(line[2]), 
                    float(line[7]), 
                    float(line[8]))
            time = float(line[4])
            temp.append(dist/time)
        else:
            break
    arr = np.array(temp)
    return np.std(arr, axis=0)
    

def one_line_average_length(session_list):
    temp = list()
    for line in session_list:
        if float(line[9]) != 0.0 and float(line[6]) != 0.0:
            dist = distance(
                    float(line[1]), 
                    float(line[2]), 
                    float(line[7]), 
                    float(line[8]))
            temp.append(dist)
        else:
            break
    try:
        sum(temp)/len(temp)
    except ZeroDivisionError:
        return 0
    else:
        return sum(temp)/len(temp)


def one_line_normal_average_length(session_list):
    temp = list()
    for line in session_list:
        if float(line[9]) != 0.0 and float(line[6]) != 0.0:
            temp.append(math.fabs(float(line[1]) - float(line[7])))
    try:
        sum(temp)/len(temp)
    except ZeroDivisionError:
        return 0
    else:
        return sum(temp)/len(temp)        


def one_line_normal_vertical_average_length(session_list):
    temp = list()
    for line in session_list:
        if float(line[9]) != 0.0 and float(line[6]) != 0.0:
            temp.append(math.fabs(float(line[2]) - float(line[8])))
    try:
        sum(temp)/len(temp)
    except ZeroDivisionError:
        return 0
    else:
        return sum(temp)/len(temp)
    
    
def one_line_std_vertical_length(session_list):
    temp = list()
    for line in session_list:
        if float(line[9]) != 0.0 and float(line[6]) != 0.0:
            temp.append(math.fabs(float(line[2]) - float(line[8])))
        else:
            break
    arr = np.array(temp)
    return np.std(arr, axis=0)


def one_line_std_horizontal_length(session_list):
    temp = list()
    for line in session_list:
        if float(line[9]) != 0.0 and float(line[6]) != 0.0:
            dist = distance(
                    float(line[1]), 
                    float(line[2]), 
                    float(line[7]), 
                    float(line[8]))
            temp.append(dist)
        else:
            break
    arr = np.array(temp)
    return np.std(arr, axis=0)


def one_line_std_sinus(session_list):
    temp = list()
    for line in session_list:
        if float(line[9]) != 0.0 and float(line[6]) != 0.0:
            hypotenuse = distance(
                    float(line[1]), 
                    float(line[2]), 
                    float(line[7]), 
                    float(line[8]))
            catheter = float(line[2]) - float(line[8])
            temp.append(catheter/hypotenuse)
        else:
            break
#    try:
#        sum(temp)/len(temp)
#    except ZeroDivisionError:
#        return 0
#    else:
#        return sum(temp)/len(temp)
#    return np.std(np.array(temp), axis=0)
    arr = np.array(temp)
    return np.std(arr, axis=0)


def inter_line_dist(session_list):
    temp = 0
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0 and float(session_list[index][6]) != 0:
            temp += distance(
                    float(session_list[index - 1][7]), 
                    float(session_list[index - 1][8]),
                    float(session_list[index][1]),
                    float(session_list[index][2]))
        else:
             break   
    return temp


def inter_line_speed(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            dist = distance(
                    float(session_list[index - 1][7]), 
                    float(session_list[index - 1][8]),
                    float(session_list[index][1]),
                    float(session_list[index][2]))
            time = float(session_list[index][0]) - float(session_list[index - 1][0])
            temp.append(dist/time)
        else:
            break
    try:
        sum(temp)/len(temp)
    except ZeroDivisionError:
        return 0
    else:
        return sum(temp)/len(temp)


def inter_line_std_speed(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            dist = distance(
                    float(session_list[index - 1][7]), 
                    float(session_list[index - 1][8]),
                    float(session_list[index][1]),
                    float(session_list[index][2]))
            time = float(session_list[index][0]) - float(session_list[index - 1][0])
            temp.append(dist/time)
        else:
            break
    arr = np.array(temp)
    return np.std(arr, axis=0)


def inter_line_average_length(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            dist = distance(
                    float(session_list[index - 1][7]), 
                    float(session_list[index - 1][8]),
                    float(session_list[index][1]),
                    float(session_list[index][2]))
            temp.append(dist)
    try:
        sum(temp)/len(temp)
    except ZeroDivisionError:
        return 0
    else:
        return sum(temp)/len(temp)


def inter_line_normal_average_length(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            temp.append(math.fabs(float(session_list[index - 1][7]) - float(session_list[index][1])))
    try:
        sum(temp)/len(temp)
    except ZeroDivisionError:
        return 0
    else:
        return sum(temp)/len(temp)


def inter_line_normal_vertical_average_length(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            temp.append(math.fabs(float(session_list[index - 1][8]) - float(session_list[index][2])))
    try:
        sum(temp)/len(temp)
    except ZeroDivisionError:
        return 0
    else:
        return sum(temp)/len(temp)


def inter_line_std_sinus(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            hypotenuse = distance(
                    float(session_list[index - 1][7]), 
                    float(session_list[index - 1][8]),
                    float(session_list[index][1]),
                    float(session_list[index][2]))
            catheter = float(session_list[index - 1][8]) - float(session_list[index][2])
            temp.append(catheter/hypotenuse)
    arr = np.array(temp)
    return np.std(arr, axis=0)


def total_time_calc(session_list):
    temp = 0
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            temp += (float(session_list[index][0]) - float(session_list[index - 1][0]))
        else:
            break
    return temp


def fix_time_calc(session_list):
    temp = 0
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            temp += math.fabs(float(session_list[index][4]) - float(session_list[index - 1][4]))
        else:
            break
    return temp


def fix_time_std(session_list):
    temp = 0
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            temp += math.fabs(float(session_list[index][4]) - float(session_list[index - 1][4]))
        else:
            break
    arr = np.array(temp)
    return np.std(arr, axis=0)

def nominal_session_time(session_list):
    return math.fabs(float(session_list[-1][0]) - float(session_list[0][0]))

        
def session_collapsing(session_list):
    horizontal_distance = one_line_dist(session_list)
    vertical_distance = inter_line_dist(session_list)
    average_horizontal_distance = one_line_average_length(session_list)
    average_verticall_distance = inter_line_average_length(session_list)
    total_time = total_time_calc(session_list)#float(session_list[-1][7])
    fix_time_change = fix_time_calc(session_list)
    inline_speed = one_line_speed(session_list)
    between_line_speed = inter_line_speed(session_list)
    inline_normal_horizontal_distance = one_line_normal_average_length(session_list)
    inline_normal_vertical_distance = one_line_normal_vertical_average_length(session_list)
    interline_normal_horizontal_distance = inter_line_normal_average_length(session_list)
    session_time = nominal_session_time(session_list)
    inter_line_sinus = inter_line_std_sinus(session_list)
    one_line_sinus = one_line_std_sinus(session_list)
    inline_std_speed = one_line_std_speed(session_list)
    interline_std_speed = inter_line_std_speed(session_list)
    inline_vertical_std = one_line_std_vertical_length(session_list)
    inline_horizontal_std = one_line_std_horizontal_length(session_list)
    fixation_interval_std = fix_time_std(session_list)
#    total_blinks = math.fabs(float(session_list[0][12]) - float(session_list[-1][12]))
    return [horizontal_distance, 
            vertical_distance,
            average_horizontal_distance,
            average_verticall_distance,
            total_time, 
            fix_time_change, 
            inline_speed,
            between_line_speed,
            inline_normal_horizontal_distance,
            interline_normal_horizontal_distance,
            inline_normal_vertical_distance,
            session_time,
            inter_line_sinus,
            one_line_sinus,
            inline_std_speed,
            interline_std_speed,
            inline_vertical_std,
            inline_horizontal_std,
            fixation_interval_std]


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
                                           'X6', 
                                           'X7', 
                                           'X8',
                                           'X9',
                                           'X10',
                                           'X11',
                                           'X12',
                                           'X13',
                                           'X14',
                                           'X15',
                                           'X16',
                                           'X17',
                                           'X18'])
    return df


def soft_voting(df_res, y):
    clf2 = ensemble.RandomForestClassifier(n_estimators=5000,
                                           max_depth=9,
                                           max_features=1, 
                                           random_state=42, 
                                           n_jobs=-1)
    clf3 = ensemble.GradientBoostingClassifier(n_estimators=5000, 
                                              learning_rate=0.001, 
                                              max_depth=9, 
                                              random_state=42)
    clf1 = GaussianNB(priors=None)
    clf4 = SGDClassifier(max_iter=10000,
                         alpha=0.001,
                         tol=1e-4, 
                         shuffle=True, 
                         penalty='l2', 
                         loss='log')
    clf5 = DecisionTreeClassifier(max_depth=7)
    clf6 = SVC(kernel="linear", C=0.025, probability=True)
    clf7 = SVC(gamma=2, C=1, probability=True)
    clf8 = LogisticRegression(C=1e5)
    clf9 = MLPClassifier(
            hidden_layer_sizes=(35, 30, 25, 20, 15, 10), 
            alpha=0.0001,
            max_iter=3000,
            activation='logistic')
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
        scores = cross_val_score(clf, df_res, y, cv=10, n_jobs=-1, scoring='roc_auc')
        print("ROC_AUC scoring: %0.3f (+/- %0.3f) [%s]" % (scores.mean(), scores.std(), label))


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
    CN_list = [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000, 100000]
    MAX_iter = [10, 100, 1000, 10000, 100000]
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
#    frame = read_data(url_data, file_name)
    final_list = main(data_path)
    cleaned_final = final_list_clean(final_list)
    total_dataframe = list_to_frame(cleaned_final)

    y = total_dataframe['Target']
    X = total_dataframe.drop('Target', 1)

    
##    X_proc = (X - X.mean()) / X.std()
#    
#    poly = preprocessing.PolynomialFeatures(interaction_only=True)
#    poly.fit(X)
#    X_proc0 = poly.transform(X)
#
##    min_max_scaler = preprocessing.MinMaxScaler()
##    X_proc = min_max_scaler.fit_transform(X)
#
##    standart_scaler = preprocessing.StandardScaler() #0.629
##    X_proc = standart_scaler.fit_transform(X_proc0)
#    
#    quantily = preprocessing.QuantileTransformer(output_distribution='uniform') #0.635
#    X_proc = quantily.fit_transform(X_proc0)
#    
##    quantily = preprocessing.QuantileTransformer(output_distribution='normal') #0.634
##    X_proc = quantily.fit_transform(X_proc0)
#    
#    X_proc = np.delete(X_proc, np.s_[0:1], axis=1)
#    
##    normalizer = preprocessing.Normalizer() #0.628
##    X_proc = normalizer.fit_transform(X_proc0)
#    
##    X_proc = preprocessing.RobustScaler(quantile_range=(25, 75)).fit_transform(X) #0.626
    X_proc = X
    
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, 
                                                        test_size=0.3, 
                                                        random_state=42, 
                                                        shuffle=True)
    soft_voting(X_train, y_train)
    
    print('*******************')
    print('\n')
    
    standart_scaler = preprocessing.StandardScaler() #0.629
    X_proc = standart_scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, 
                                                        test_size=0.3, 
                                                        random_state=42, 
                                                        shuffle=True)    
    soft_voting(X_train, y_train)    
#    select_C_for_LogReg(X_train, y_train)
    

#parameters={
#'learning_rate': ["constant", "invscaling", "adaptive"],
#'hidden_layer_sizes': [(100,1), (100,2), (100,3)],
#'alpha': [10.0 ** -np.arange(1, 7)],
#'activation': ["logistic", "relu", "Tanh"]
#}
#
#clf = gridSearchCV(estimator=MLPClassifier,param_grid=parameters,n_jobs=-1,verbose=2,cv=10)
#https://datascience.stackexchange.com/questions/19768/how-to-implement-pythons-mlpclassifier-with-gridsearchcv
#http://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html


    
#    female_file_to_work, male_file_to_work, file_number = read_files(data_path)
#    data_list = [female_file_to_work, male_file_to_work]
#    for data in data_list:
#        if data_list.index(data) == 0:
#            param = 0
#            female_list = result_data_cooking(all_data_reading(data), param)
#        elif data_list.index(data) == 1:
#            param = 1
#            male_list = result_data_cooking(all_data_reading(data), param)
#    final_list = female_list + male_list
    
#    res = all_data_reading(file_names_list)
#    final_list = result_data_cooking(res)
#    frame = read_data(url_data, file_name)
    
#    clf4 = SGDClassifier(max_iter=10000, 
#                         tol=1e-4, 
#                         shuffle=True, 
#                         penalty='l2', 
#                         loss='log')
#        columns=['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
#        X1 = X[columns]