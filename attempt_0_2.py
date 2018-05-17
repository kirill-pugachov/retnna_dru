# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 17:57:31 2018

@author: Kirill
"""


import os
import pandas as pd
import csv
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#from sklearn import preprocessing
#from sklearn.grid_search import GridSearchCV
from sklearn import ensemble
#from pandas.plotting import scatter_matrix
#from contextlib import redirect_stdout
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB



#url_data = 'C:/Users/Кирилл/retnna/Data/csv_data/raw_nr/'
#file_name = 'Search_Andrii2_all_gaze.csv'

url_data = 'C:/Users/Кирилл/retnna/Data/csv_data/notreading/'
file_name = 'pct2_olga.csv' #'faces_anastasia.csv'
data_path = 'C:/Users/Кирилл/retnna/Work_data/notreading/'

#data = url_data + file_name


def read_files(data_path):
    file_number = 0
    female_file_to_work = list()
    male_file_to_work = list()
    for filename in os.listdir(data_path):
        if filename.startswith("male_"):
            file_number += 1
            male_file_to_work.append(data_path + filename)
        elif filename.startswith("female_"):
            file_number += 1
            female_file_to_work.append(data_path + filename)
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
        temp += distance(
                float(line[1]), 
                float(line[2]), 
                float(line[7]), 
                float(line[8]))
    return temp


def inter_line_dist(session_list):
    temp = 0
    index_max = len(session_list)
    for index in range(1, index_max):
        temp += distance(
                float(session_list[index - 1][7]), 
                float(session_list[index - 1][8]),
                float(session_list[index][1]),
                float(session_list[index][2]))
    return temp
        

def total_time_calc(session_list):
    pass

        
def session_collapsing(session_list):
    horizontal_distance = one_line_dist(session_list)
    vertical_distance = inter_line_dist(session_list)
    total_time = float(session_list[-1][7])
    total_blinks = math.fabs(float(session_list[0][12]) - float(session_list[-1][12]))
    return [horizontal_distance, vertical_distance, total_time, total_blinks]


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
    df = pd.DataFrame(final_list, columns=['Target','X0', 'X1', 'X2', 'X3'])
    return df


def soft_voting(df_res, y):
    clf2 = ensemble.RandomForestClassifier(n_estimators=500, random_state=11, 
                                           n_jobs=-1)
    clf3 = ensemble.GradientBoostingClassifier(n_estimators=3500, 
                                              learning_rate=0.1, max_depth=7, 
                                              random_state=11)
    clf1 = GaussianNB(priors=None)
    clf4 = SGDClassifier(max_iter=40000, tol=1e-4, shuffle=True, penalty='l2', loss='log')
    eclf = VotingClassifier(estimators=[('gau', clf1), ('rfc', clf2), 
                                        ('gbs', clf3), ('sgdc', clf4)], 
                                voting='soft', weights=[1,1,1,1])
    for clf, label in zip([clf1, clf2, clf3, clf4, eclf], 
                          ['GaussianNB', 'RandomForestClassifier', 'GradientBoosting', 'SGDClassifier', 'Ensemble']):
        scores = cross_val_score(clf, df_res, y, cv=5, n_jobs=-1, scoring='roc_auc')
        print("ROC_AUC scoring: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


if __name__ == '__main__':
#    frame = read_data(url_data, file_name)
    final_list = main(data_path)
    total_dataframe = list_to_frame(final_list)
    y = total_dataframe['Target']
    X = total_dataframe.drop('Target', 1)  
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.33, 
                                                        random_state=42, 
                                                        shuffle=True)
    soft_voting(X_train, y_train)
    
    
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