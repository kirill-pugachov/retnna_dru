# -*- coding: utf-8 -*-
"""
Created on Sat May  5 08:45:26 2018

@author: User
"""

import os
import pandas as pd
import numpy as np
import csv
import math

from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

#data_path = [
#        'C:/Users/User/retnna/Work_data/reading/',
#        'C:/Users/User/retnna/Work_data/notreading/'
#        ]


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
    temp_result_list = list()
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


#общее число строк в одной сессии
#
#
def raws_number_in_session(session_list):
    temp = 0
    for line in session_list:
        temp += 1
    return temp


#число строк с ошибками в одной сессии
#FPOGV == 0
#
def error_raws_number_in_session(session_list):
    temp = 0
    for line in session_list:
        if float(line[6]) == 0.0:
            temp += 1
    return temp


#квадратичное отклонение перемещения глаз
#в рамках одной сессии по каждой строке данных
#строчки с 0 выбрасываются
def oneline_dist_std(session_list):
    temp = list()
    for line in session_list:
        if float(line[9]) != 0.0 and float(line[6]) != 0.0:
            temp.append(distance(
                    float(line[7]),
                    float(line[8]),
                    float(line[1]),
                    float(line[2])))
        else:
            continue
    if len(temp) > 2:
        arr = np.array(temp)
        try:
            np.std(arr, axis=0)
        except:
            print(session_list[index], '\n', temp)
            return 0
        else:
            return np.std(arr, axis=0)
    else:
        return 0


#средняя длина перемещения глаз
#в рамках одной сессии по каждой строке данных
#строчки с 0 выбрасываются
def oneline_dist_mean(session_list):
    temp = list()
    for line in session_list:
        if float(line[9]) != 0.0 and float(line[6]) != 0.0:
            temp.append(distance(
                    float(line[7]),
                    float(line[8]),
                    float(line[1]),
                    float(line[2])))
        else:
            continue
    if len(temp) > 0:
        arr = np.array(temp)
        try:
            np.mean(arr)
        except:
            print(temp)
            return 0
        else:
            return np.mean(arr)
    else:
        return 0



#квадратичное отклонение синуса угла перемещения глаз
#в рамках одной сессии по каждой строке данных
#строчки с 0 выбрасываются
def one_line_std_sinus(session_list):
    temp = list()
    for line in session_list:
        if float(line[9]) != 0.0 and float(line[6]) != 0.0:
            hypotenuse = distance(
                    float(line[7]),
                    float(line[8]),
                    float(line[1]),
                    float(line[2]))
            catheter = float(line[8]) - float(line[2])
            temp.append(catheter/hypotenuse)
        else:
            continue
    if len(temp) > 2:
        arr = np.array(temp)
        try:
            np.std(arr, axis=0)
        except:
            print(temp)
            return 0
        else:
            return np.std(arr, axis=0)
    else:
        return 0


#среднее значение синуса угла перемещения глаз
#в рамках одной сессии по каждой строке данных
#строчки с 0 выбрасываются
def one_line_sinus_mean(session_list):
    temp = list()
    for line in session_list:
        if float(line[9]) != 0.0 and float(line[6]) != 0.0:
            hypotenuse = distance(
                    float(line[7]),
                    float(line[8]),
                    float(line[1]),
                    float(line[2]))
            catheter = float(line[8]) - float(line[2])
            temp.append(catheter/hypotenuse)
        else:
            continue
    if len(temp) > 0:
        arr = np.array(temp)
        try:
            np.mean(arr)
        except:
            print(temp)
            return 0
        else:
            return np.mean(arr)
    else:
        return 0


#квадратичное отклонение времени фиксации глаз
#в рамках одной сессии по каждой строке данных
#строчки с 0 выбрасываются
def fix_time_std(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            temp.append(float(session_list[index][4])) #- float(session_list[index - 1][4]))
        else:
            continue
    if len(temp) > 2:
        arr = np.array(temp)
        try:
            np.std(arr, axis=0)
        except:
            print(session_list[index], '\n', temp)
            return 0
        else:
            return np.std(arr, axis=0)
    else:
        return 0
    
    
#среднее значение времени фиксации глаз
#в рамках одной сессии по каждой строке данных
#строчки с 0 выбрасываются
def fix_time_mean(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            temp.append(float(session_list[index][4])) #- float(session_list[index - 1][4]))
        else:
            continue
    if len(temp) > 0:
        arr = np.array(temp)
        try:
            np.mean(arr)
        except:
            print(session_list[index], '\n', temp)
            return 0
        else:
            return np.mean(arr)
    else:
        return 0


#квадратичное отклонение длинны движения глаз по горизонтальной оси
#в рамках одной сессии по каждой строке данных
#строчки с 0 выбрасываются
def one_line_std_horizontal_length(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            temp.append(
                    float(session_list[index][7]) - float(session_list[index][1])
                    )
        else:
            continue
    if len(temp) > 2:
        arr = np.array(temp)
        try:
            np.std(arr, axis=0)
        except:
            print(session_list[index], '\n', temp)
            return 0
        else:
            return np.std(arr, axis=0)
    else:
        return 0


#квадратичное отклонение длинны движения глаз по вертикальной оси
#в рамках одной сессии по каждой строке данных
#строчки с 0 выбрасываются
def one_line_std_vertical_length(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
#        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
        if float(session_list[index][6]) != 0.0:
            temp.append(
                    float(session_list[index][8]) - float(session_list[index][2])
                    )
        else:
            continue
    if len(temp) > 2:
        arr = np.array(temp)
        try:
            np.std(arr, axis=0)
        except:
            print(session_list[index], '\n', temp)
            return 0
        else:
            return np.std(arr, axis=0)
    else:
        return 0


#квадратичное отклонение скорости движения глаз по экрану
#в рамках одной сессии по каждой строке данных
#строчки с 0 выбрасываются
def one_line_std_speed(session_list):
    temp = list()
    for line in session_list:
        if float(line[6]) != 0.0 and float(line[9]) != 0.0:
#        if float(line[6]) != 0.0:
            dist = distance(
                    float(line[7]),
                    float(line[8]),
                    float(line[1]),
                    float(line[2]))
            time = float(line[4])
            temp.append(dist/time)
        else:
            continue
    if len(temp) > 2:
        arr = np.array(temp)
        try:
            np.std(arr, axis=0)
        except:
            return 0
        else:
            return np.std(arr, axis=0)
    else:
        return 0
    
    
#среднее значение скорости движения глаз по экрану
#в рамках одной сессии по каждой строке данных
#строчки с 0 выбрасываются
def one_line_speed_mean(session_list):
    temp = list()
    for line in session_list:
        if float(line[6]) != 0.0 and float(line[9]) != 0.0:
            dist = distance(
                    float(line[7]),
                    float(line[8]),
                    float(line[1]),
                    float(line[2]))
            time = float(line[4])
            temp.append(dist/time)
        else:
            continue
    if len(temp) > 0:
        arr = np.array(temp)
        try:
            np.mean(arr)
        except:
            return 0
        else:
            return np.mean(arr)
    else:
        return 0


#квадратичное отклонение по оси Y движения глаз по экрану
#в рамках одной сессии по каждой строке данных
#строчки с 0 выбрасываются
def one_line_std_Y_speed(session_list):
    temp = list()
    for line in session_list:
        if float(line[6]) != 0.0 and float(line[9]) != 0.0:
#        if float(line[6]) != 0.0:
            dist = float(line[8]) - float(line[2])
            time = float(line[4])
            temp.append(dist/time)
        else:
            continue
    if len(temp) > 2:
        arr = np.array(temp)
        try:
            np.std(arr, axis=0)
        except:
            return 0
        else:
            return np.std(arr, axis=0)
    else:
        return 0


#средняя скорость по оси Y движения глаз по экрану
#в рамках одной сессии по каждой строке данных
#строчки с 0 выбрасываются
def one_line_Y_speed_mean(session_list):
    temp = list()
    for line in session_list:
        if float(line[6]) != 0.0 and float(line[9]) != 0.0:
#        if float(line[6]) != 0.0:
            dist = float(line[8]) - float(line[2])
            time = float(line[4])
            temp.append(dist/time)
        else:
            continue
    if len(temp) > 0:
        arr = np.array(temp)
        try:
            np.mean(arr)
        except:
            return 0
        else:
            return np.mean(arr)
    else:
        return 0
    

#квадратичное отклонение скорости по оси X движения глаз по экрану
#в рамках одной сессии по каждой строке данных
#строчки с 0 выбрасываются
def one_line_std_X_speed(session_list):
    temp = list()
    for line in session_list:
        if float(line[6]) != 0.0 and float(line[9]) != 0.0:
#        if float(line[6]) != 0.0:
            dist = float(line[7]) - float(line[1])
            time = float(line[4])
            temp.append(dist/time)
        else:
            continue
    if len(temp) > 2:
        arr = np.array(temp)
        try:
            np.std(arr, axis=0)
        except:
            return 0
        else:
            return np.std(arr, axis=0)
    else:
        return 0


#Среднее значение скорости по оси X движения глаз по экрану
#в рамках одной сессии по каждой строке данных
#строчки с 0 выбрасываются
def one_line_X_speed_mean(session_list):
    temp = list()
    for line in session_list:
        if float(line[6]) != 0.0 and float(line[9]) != 0.0:
#        if float(line[6]) != 0.0:
            dist = float(line[7]) - float(line[1])
            time = float(line[4])
            temp.append(dist/time)
        else:
            continue
    if len(temp) > 0:
        arr = np.array(temp)
        try:
            np.mean(arr)
        except:
            return 0
        else:
            return np.mean(arr)
    else:
        return 0
    
    

#квадратичное отклонение скорости перемещения между строками
#в рамках одной сессии по каждой строке данных
#строчки с 0 выбрасываются
def interline_std_speed(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            if float(session_list[index - 1][9]) != 0.0 and float(session_list[index - 1][6]) != 0.0:
                dist = distance(
                        float(session_list[index - 1][7]),
                        float(session_list[index - 1][8]),
                        float(session_list[index][1]),
                        float(session_list[index][2]))
                time = float(session_list[index][0]) - float(session_list[index - 1][4]) - float(session_list[index - 1][0])
                temp.append(dist/time)
            else:
                continue
        else:
            continue
    if len(temp) > 2:
        arr = np.array(temp)
        try:
            np.std(arr, axis=0)
        except:
            return 0
        else:
            return np.std(arr, axis=0)
    else:
        return 0


#среднее значение скорости перемещения между строками
#в рамках одной сессии по каждой паре строке данных
#строчки с 0 выбрасываются и пару не образуют
def interline_speed_mean(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            if float(session_list[index - 1][9]) != 0.0 and float(session_list[index - 1][6]) != 0.0:
                dist = distance(
                        float(session_list[index - 1][7]),
                        float(session_list[index - 1][8]),
                        float(session_list[index][1]),
                        float(session_list[index][2]))
                time = float(session_list[index][0]) - float(session_list[index - 1][4]) - float(session_list[index - 1][0])
                temp.append(dist/time)
            else:
                continue
        else:
            continue
    if len(temp) > 0:
        arr = np.array(temp)
        try:
            np.mean(arr)
        except:
            return 0
        else:
            return np.mean(arr)
    else:
        return 0
    

#квадратичное отклонение пройденного расстояние при перемещении между строками
#в рамках одной сессии от крайней правой точки строки 1 до начальной точки
#строчки 2, только последовательные пары, все с 0 выбрасываются
def interline_std_distance(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            if float(session_list[index - 1][9]) != 0.0 and float(session_list[index - 1][6]) != 0.0:
                dist = distance(
                        float(session_list[index - 1][7]),
                        float(session_list[index - 1][8]),
                        float(session_list[index][1]),
                        float(session_list[index][2]))
                temp.append(dist)
            else:
                continue
        else:
            continue
    if len(temp) > 2:
        arr = np.array(temp)
        try:
            np.std(arr, axis=0)
        except:
            return 0
        else:
            return np.std(arr, axis=0)
    else:
        return 0


#среднее значение пройденного расстояние при перемещении между строками
#в рамках одной сессии от крайней правой точки строки 1 до начальной точки
#строчки 2, только последовательные пары, все с 0 выбрасываются
def interline_distance_mean(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            if float(session_list[index - 1][9]) != 0.0 and float(session_list[index - 1][6]) != 0.0:
                dist = distance(
                        float(session_list[index - 1][7]),
                        float(session_list[index - 1][8]),
                        float(session_list[index][1]),
                        float(session_list[index][2]))
                temp.append(dist)
            else:
                continue
        else:
            continue
    if len(temp) > 0:
        arr = np.array(temp)
        try:
            np.mean(arr)
        except:
            return 0
        else:
            return np.mean(arr)
    else:
        return 0


#квадратичное отклонение вертикального расстояние при перемещении между строками
#в рамках одной сессии от Y крайней правой точки строки 1 до Y начальной точки
#строчки 2, только последовательные пары, все с 0 выбрасываются
def interline_std_vertical_distance(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            if float(session_list[index - 1][9]) != 0.0 and float(session_list[index - 1][6]) != 0.0:
                dist = float(session_list[index - 1][2]) - float(session_list[index][8])
                temp.append(dist)
            else:
                continue
        else:
            continue
    if len(temp) > 2:
        arr = np.array(temp)
        try:
            np.std(arr, axis=0)
        except:
            return 0
        else:
            return np.std(arr, axis=0)
    else:
        return 0


#среднее значение вертикального расстояние при перемещении между строками
#в рамках одной сессии от Y крайней правой точки строки 1 до Y начальной точки
#строчки 2, только последовательные пары, все с 0 выбрасываются
def interline_vertical_distance_mean(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            if float(session_list[index - 1][9]) != 0.0 and float(session_list[index - 1][6]) != 0.0:
                dist = float(session_list[index - 1][2]) - float(session_list[index][8])
                temp.append(dist)
            else:
                continue
        else:
            continue
    if len(temp) > 0:
        arr = np.array(temp)
        try:
            np.mean(arr)
        except:
            return 0
        else:
            return np.mean(arr)
    else:
        return 0
    


#квадратичное отклонение горизонтального расстояние при перемещении между строками
#в рамках одной сессии от Х крайней правой точки строки 1 до Х начальной точки
#строчки 2, только последовательные пары, все с 0 выбрасываются
def interline_std_horizont_distance(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            if float(session_list[index - 1][9]) != 0.0 and float(session_list[index - 1][6]) != 0.0:
                dist = float(session_list[index - 1][1]) - float(session_list[index][7])
                temp.append(dist)
            else:
                continue
        else:
            continue
    if len(temp) > 2:
        arr = np.array(temp)
        try:
            np.std(arr, axis=0)
        except:
            return 0
        else:
            return np.std(arr, axis=0)
    else:
        return 0


#среднее значение горизонтального расстояние при перемещении между строками
#в рамках одной сессии от Х крайней правой точки строки 1 до Х начальной точки
#строчки 2, только последовательные пары, все с 0 выбрасываются
def interline_horizont_distance_mean(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            if float(session_list[index - 1][9]) != 0.0 and float(session_list[index - 1][6]) != 0.0:
                dist = float(session_list[index - 1][1]) - float(session_list[index][7])
                temp.append(dist)
            else:
                continue
        else:
            continue
    if len(temp) > 0:
        arr = np.array(temp)
        try:
            np.mean(arr)
        except:
            return 0
        else:
            return np.mean(arr)
    else:
        return 0


#квадратичное отклонение скорости вертикального движения при перемещении между строками
#в рамках одной сессии от Y крайней правой начальной точки строки 1 до Y начальной точки
#строчки 2, только последовательные пары, все с 0 выбрасываются
def interline_std_vertical_speed(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            if float(session_list[index - 1][9]) != 0.0 and float(session_list[index - 1][6]) != 0.0:
                dist = float(session_list[index - 1][2]) - float(session_list[index][8])
                time = float(session_list[index][0]) - float(session_list[index - 1][4]) - float(session_list[index - 1][0])
                temp.append(dist/time)
            else:
                continue
        else:
            continue
    if len(temp) > 2:
        arr = np.array(temp)
        try:
            np.std(arr, axis=0)
        except:
            return 0
        else:
            return np.std(arr, axis=0)
    else:
        return 0


#среднее значение скорости вертикального движения при перемещении между строками
#в рамках одной сессии от Y крайней правой начальной точки строки 1 до Y начальной точки
#строчки 2, только последовательные пары, все с 0 выбрасываются
def interline_vertical_speed_mean(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            if float(session_list[index - 1][9]) != 0.0 and float(session_list[index - 1][6]) != 0.0:
                dist = float(session_list[index - 1][2]) - float(session_list[index][8])
                time = float(session_list[index][0]) - float(session_list[index - 1][4]) - float(session_list[index - 1][0])
                temp.append(dist/time)
            else:
                continue
        else:
            continue
    if len(temp) > 0:
        arr = np.array(temp)
        try:
            np.mean(arr)
        except:
            return 0
        else:
            return np.mean(arr)
    else:
        return 0


#квадратичное отклонение скорости горизонтального движения при перемещении между строками
#в рамках одной сессии от Y крайней правой начальной точки строки 1 до Y начальной точки
#строчки 2, только последовательные пары, все с 0 выбрасываются
def interline_std_horizontal_speed(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            if float(session_list[index - 1][9]) != 0.0 and float(session_list[index - 1][6]) != 0.0:
                dist = float(session_list[index - 1][1]) - float(session_list[index][7])
                time = float(session_list[index][0]) - float(session_list[index - 1][4]) - float(session_list[index - 1][0])
                temp.append(dist/time)
            else:
                continue
        else:
            continue
    if len(temp) > 2:
        arr = np.array(temp)
        try:
            np.std(arr, axis=0)
        except:
            return 0
        else:
            return np.std(arr, axis=0)
    else:
        return 0


#квадратичное отклонение скорости горизонтального движения при перемещении между строками
#в рамках одной сессии от Y крайней правой начальной точки строки 1 до Y начальной точки
#строчки 2, только последовательные пары, все с 0 выбрасываются
def interline_horizontal_speed_mean(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            if float(session_list[index - 1][9]) != 0.0 and float(session_list[index - 1][6]) != 0.0:
                dist = float(session_list[index - 1][1]) - float(session_list[index][7])
                time = float(session_list[index][0]) - float(session_list[index - 1][4]) - float(session_list[index - 1][0])
                temp.append(dist/time)
            else:
                continue
        else:
            continue
    if len(temp) > 0:
        arr = np.array(temp)
        try:
            np.mean(arr)
        except:
            return 0
        else:
            return np.mean(arr)
    else:
        return 0


#общая длинна пути глаз на успешных отрезках сессии
#сумма длинны пути по строке и между строк
#
def success_total_length(session_list):
    temp = 0
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            if float(session_list[index - 1][9]) != 0.0 and float(session_list[index - 1][6]) != 0.0:
                temp += distance(
                        float(session_list[index - 1][7]),
                        float(session_list[index - 1][8]),
                        float(session_list[index][1]),
                        float(session_list[index][2]))
                temp += distance(
                        float(session_list[index][1]),
                        float(session_list[index][2]),
                        float(session_list[index][7]),
                        float(session_list[index][8]))
            else:
                continue
        else:
            continue                
    return temp


#квадратичное отклонение синуса угла перемещения глаз
#между строками одной сессии по каждой валидной паре строк данных
#строчки с 0 выбрасываются
def interline_std_sinus(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            if float(session_list[index - 1][9]) != 0.0 and float(session_list[index - 1][6]) != 0.0:
                hypotenuse = distance(
                        float(session_list[index - 1][7]),
                        float(session_list[index - 1][8]),
                        float(session_list[index][1]),
                        float(session_list[index][2]))
                catheter = float(session_list[index - 1][8]) - float(session_list[index][2])
                temp.append(catheter/hypotenuse)
            else:
                continue
        else:
            continue
    if len(temp) > 2:
        arr = np.array(temp)
        try:
            np.std(arr, axis=0)
        except:
            print(session_list[index], '\n', temp)
            return 0
        else:
            return np.std(arr, axis=0)
    else:
        return 0


#квадратичное отклонение синуса угла перемещения глаз
#между строками одной сессии по каждой валидной паре строк данных
#строчки с 0 выбрасываются
def interline_sinus_mean(session_list):
    temp = list()
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            if float(session_list[index - 1][9]) != 0.0 and float(session_list[index - 1][6]) != 0.0:
                hypotenuse = distance(
                        float(session_list[index - 1][7]),
                        float(session_list[index - 1][8]),
                        float(session_list[index][1]),
                        float(session_list[index][2]))
                catheter = float(session_list[index - 1][8]) - float(session_list[index][2])
                temp.append(catheter/hypotenuse)
            else:
                continue
        else:
            continue
    if len(temp) > 0:
        arr = np.array(temp)
        try:
            np.mean(arr)
        except:
            print(session_list[index], '\n', temp)
            return 0
        else:
            return np.mean(arr)
    else:
        return 0


#сумма горизонтальных расстояний в рамках одной сессии
#до первой ошибки в замерах/фиксации движения глаз
#
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


#сумма пройденных межстрочных расстояний в рамках одной сессии
#до первой ошибки в замерах/фиксации движения глаз
#
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

#сумма осевого времени - по временной метке, в рамках одной сессии
#до первой ошибки в замерах/фиксации движения глаз
#
def total_time_calc(session_list):
    temp = 0
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            temp += (float(session_list[index][0]) - float(session_list[index - 1][0]))
        else:
            break
    return temp

#общая длинна одной сессии
#
#
def nominal_session_time(session_list):
    return math.fabs(float(session_list[-1][0]) - float(session_list[0][0]))


#сумма времени фиксации - по временной метке, в рамках одной сессии
#до первой ошибки в замерах/фиксации движения глаз
#
def fix_time_calc(session_list):
    temp = 0
    index_max = len(session_list)
    for index in range(1, index_max):
        if float(session_list[index][9]) != 0.0 and float(session_list[index][6]) != 0.0:
            temp += math.fabs(float(session_list[index][4]) - float(session_list[index - 1][4]))
        else:
            break
    return temp


def session_collapsing(session_list):
    
    horizontal_distance = one_line_dist(session_list)
    vertical_distance = inter_line_dist(session_list)

    inline_dist_std = oneline_dist_std(session_list)
    inline_dist_mean = oneline_dist_mean(session_list)
    
    total_time = total_time_calc(session_list)
    fix_time_change = fix_time_calc(session_list)
    
    inline_sin_std = one_line_std_sinus(session_list)
    inline_sin_mean = one_line_sinus_mean(session_list)
    session_fix_time_std = fix_time_std(session_list)
    session_fix_time_mean = fix_time_mean(session_list)
    inline_h_move_length_std = one_line_std_horizontal_length(session_list)
    inline_v_move_length_std = one_line_std_vertical_length(session_list)
    inline_speed_std = one_line_std_speed(session_list)
    inline_speed_mean = one_line_speed_mean(session_list)
    inline_Y_speed_std = one_line_std_Y_speed(session_list)
    inline_Y_speed_mean = one_line_Y_speed_mean(session_list)
    inline_X_speed_std = one_line_std_X_speed(session_list)
    inline_X_speed_mean = one_line_X_speed_mean(session_list)
    raws_number = raws_number_in_session(session_list)
    error_raws_number = error_raws_number_in_session(session_list)
    success_length = success_total_length(session_list)
    inter_raws_speed_std = interline_std_speed(session_list)
    inter_raws_speed_mean = interline_speed_mean(session_list)
    inter_raws_distance_std = interline_std_distance(session_list)
    inter_raws_distance_mean = interline_distance_mean(session_list)
    inter_raws_v_move_length_std = interline_std_vertical_distance(session_list)
    inter_raws_v_move_length_mean = interline_vertical_distance_mean(session_list)
    inter_raws_h_move_length_std = interline_std_horizont_distance(session_list)
    inter_raws_h_move_length_mean = interline_horizont_distance_mean(session_list)
    inter_raws_v_speed_std = interline_std_vertical_speed(session_list)
    inter_raws_v_speed_mean = interline_vertical_speed_mean(session_list)
    inter_raws_h_speed_std = interline_std_horizontal_speed(session_list)
    inter_raws_h_speed_mean = interline_horizontal_speed_mean(session_list)
    inter_raws_sinus_std = interline_std_sinus(session_list)
    inter_raws_sinus_mean = interline_sinus_mean(session_list)

    return [
            horizontal_distance,
            vertical_distance,
            inline_dist_std,
            inline_dist_mean,
            total_time,
            fix_time_change,
            inline_sin_std,
            inline_sin_mean,
            session_fix_time_std,
            session_fix_time_mean,
            inline_h_move_length_std,
            inline_v_move_length_std,
            inline_speed_std,
            inline_speed_mean,
            inline_Y_speed_std,
            inline_Y_speed_mean,
            inline_X_speed_std,
            inline_X_speed_mean,
            raws_number,
            error_raws_number,
            success_length,
            inter_raws_speed_std,
            inter_raws_speed_mean,
            inter_raws_distance_std,
            inter_raws_distance_mean,
            inter_raws_v_move_length_std,
            inter_raws_v_move_length_mean,
            inter_raws_h_move_length_std,
            inter_raws_h_move_length_mean,
            inter_raws_v_speed_std,
            inter_raws_v_speed_mean,
            inter_raws_h_speed_std,
            inter_raws_h_speed_mean,
            inter_raws_sinus_std,
            inter_raws_sinus_mean
            ]


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
    df = pd.DataFrame(final_list, columns=['Target',
                                           'horizontal_distance',
                                           'vertical_distance',
                                           'inline_dist_std',
                                           'inline_dist_mean',
                                           'total_time',
                                           'fix_time_change',
                                           'inline_sin_std',
                                           'inline_sin_mean',
                                           'session_fix_time_std',
                                           'session_fix_time_mean',
                                           'inline_h_move_length_std',
                                           'inline_v_move_length_std',
                                           'inline_speed_std',
                                           'inline_speed_mean',
                                           'inline_Y_speed_std',
                                           'inline_Y_speed_mean',
                                           'inline_X_speed_std',
                                           'inline_X_speed_mean',
                                           'raws_number',
                                           'error_raws_number',
                                           'success_length',
                                           'inter_raws_speed_std',
                                           'inter_raws_speed_mean',
                                           'inter_raws_distance_std',
                                           'inter_raws_distance_mean',
                                           'inter_raws_v_move_length_std',
                                           'inter_raws_v_move_length_mean',
                                           'inter_raws_h_move_length_std',
                                           'inter_raws_h_move_length_mean',
                                           'inter_raws_v_speed_std',
                                           'inter_raws_v_speed_mean',
                                           'inter_raws_h_speed_std',
                                           'inter_raws_h_speed_mean',
                                           'inter_raws_sinus_std',
                                           'inter_raws_sinus_mean'
                                           ])
    return df


def classifiers_evaluation(df_res, y):

    classifiers = [
                KNeighborsClassifier(3, n_jobs=-1),
                SVC(probability=True),
#                DecisionTreeClassifier(),
                ensemble.RandomForestClassifier(n_jobs=-1),
                ensemble.AdaBoostClassifier(),
                ensemble.GradientBoostingClassifier(),
                GaussianNB(),
                LinearDiscriminantAnalysis(),
#                QuadraticDiscriminantAnalysis(),
                LogisticRegression(n_jobs=-1),
                MLPClassifier(),
                SGDClassifier(loss='log', max_iter=10000, alpha=0.0001),
                LogisticRegressionCV(n_jobs=-1)
#                GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=-1)
                ]

    log_cols = ["Classifier", "ROC_AUC score"]
    log = pd.DataFrame(columns=log_cols)

    quantile = preprocessing.QuantileTransformer() #1
    X = quantile.fit_transform(df_res)

#    minmax = preprocessing.MinMaxScaler()
#    X = minmax.fit_transform(df_res)
    
#    maxabs = preprocessing.MaxAbsScaler()
#    X = maxabs.fit_transform(df_res)

#    robust = preprocessing.RobustScaler()
#    X = robust.fit_transform(df_res)
    
#    normal = preprocessing.Normalizer()
#    X = normal.fit_transform(df_res)
    
#    standart_scaler = preprocessing.StandardScaler()
#    X = standart_scaler.fit_transform(df_res)
    
#    X = np.array(df_res)

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    acc_dict = {}

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for clf in classifiers:
            name = clf.__class__.__name__
            clf.fit(X_train, y_train)
            train_predictions = clf.predict(X_test)
#            acc = accuracy_score(y_test, train_predictions)
            acc = roc_auc_score(y_test, train_predictions)

            if name in acc_dict:
                acc_dict[name] += acc
            else:
                acc_dict[name] = acc

    for clf in acc_dict:
        acc_dict[clf] = acc_dict[clf] / 5.0
        log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
        log = log.append(log_entry)

    print(acc_dict)
    print(log)


def short_soft_voting(df_res, y):
    clf2 = ensemble.RandomForestClassifier(n_estimators=750,
                                           max_depth=15,
                                           criterion='entropy',
                                           random_state=42,
                                           n_jobs=-1)
    clf3 = ensemble.GradientBoostingClassifier(n_estimators=1000,
                                              learning_rate=0.01,
                                              max_depth=16,
                                              random_state=42)
    clf5 = DecisionTreeClassifier(max_depth=15)
    clf7 = SVC(gamma=2, C=0.1, probability=True)
    clf8 = LogisticRegression(C=1e5, solver='newton-cg', max_iter=12000)
    clf9 = MLPClassifier(hidden_layer_sizes=(35, 17, 8), alpha=0.00001, max_iter=10000, activation='logistic')
    clf10 = ensemble.AdaBoostClassifier()
    clf13 = GaussianProcessClassifier(1.0 * RBF(1.0))
    eclf = VotingClassifier(estimators=[('rfc', clf2),
                                        ('gbs', clf3),
                                        ('dtc', clf5),
                                        ('svm_gamma', clf7), ('LogReg', clf8),
                                        ('Neuro_mlp', clf9), ('AdaBoost', clf10),
                                        ('GausPC', clf13)],
                                voting='soft',
                                weights=[1, 1, 1, 1, 1, 1, 1, 1])
    for clf, label in zip([clf2, clf3, clf5, clf7, clf8, clf9, clf10, clf13, eclf],
                          ['RandomForestClassifier',
                           'GradientBoosting',
                           'DecisionTreeClassifier',
                           'SVC_gamma',
                           'LogReg',
                           'Neuro_MLP',
                           'AdaBoost',
                           'GausPC',
                           'Ensemble']):
        scores = cross_val_score(
                clf, df_res, y, cv=5, n_jobs=-1, scoring='roc_auc')
        print(
                "ROC_AUC scoring: %0.3f (+/- %0.3f) [%s]" % (
                        scores.mean(), scores.std(), label))


def data_preprocess(X):
    poly = preprocessing.PolynomialFeatures(interaction_only=True)
    poly.fit(X)
    X_temp = poly.transform(X)
    standart_scaler = preprocessing.StandardScaler()
    return standart_scaler.fit_transform(X_temp)


def final_list_clean(final_list):
    result = list()
    for line in final_list:
        if float(line[1]) != 0.0:
            if float(line[2]) != 0.0:
                if float(line[5]) != 0.0:
                    if float(line[6]) != 0.0:
                        result.append(line)
    return result


def select_params_for_RandomForestClassifier(df_res, y):
    estimators_list = [300, 500, 750, 1000, 1200, 1500, 2000]
    depth_list = [8, 9, 10, 11, 12, 13, 14, 15, 16]
    criterion_list = ['gini', 'entropy']
    for estim in estimators_list:
        for depth in depth_list:
            for crit in criterion_list:
                label = 'RandomForestClassifier'
                clf = ensemble.RandomForestClassifier(n_estimators=estim, max_depth=depth, criterion=crit)
                scores = cross_val_score(clf, df_res, y, cv=10, n_jobs=-1, scoring='roc_auc')
                print("ROC_AUC scoring: %0.3f (+/- %0.3f) [%s] [%s] [%s] [%s]" % (scores.mean(), 
                                               scores.std(), 
                                               label, 
                                               estim, 
                                               depth, 
                                               crit))


def select_params_for_AdaBoostClassifier(df_res, y):
    estimators_list = [20, 40, 60, 80, 100, 200, 500, 1000]
    depth_list = [0.1, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
    label = 'RandomForestClassifier'
    for estim in estimators_list:
        for depth in depth_list:
            clf = ensemble.AdaBoostClassifier(n_estimators=estim, learning_rate=depth, algorithm='SAMME')
            scores = cross_val_score(clf, df_res, y, cv=10, n_jobs=-1, scoring='roc_auc')
            print("ROC_AUC scoring: %0.3f (+/- %0.3f) [%s] [%s] [%s]" % (scores.mean(), 
                                               scores.std(), 
                                               label, 
                                               estim, 
                                               depth))


def select_params_for_MLPClassifier(df_res, y):
    hidden_layers_list = [(150, 35, 8), (35, 150, 8), (35, 18, 4), (35, 18, 9), (35, 20, 10, 5)]
    alpha_list = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
    it_r_list = [200, 500, 1000, 5000, 10000]
    label = 'MLPClassifier'    
    for la in hidden_layers_list:
        for al in alpha_list:
            for it in it_r_list:
                clf = MLPClassifier(hidden_layer_sizes=la, alpha=al, max_iter=it, activation='logistic')
                scores = cross_val_score(clf, df_res, y, cv=10, n_jobs=-1, scoring='roc_auc')
                print("ROC_AUC scoring: %0.3f (+/- %0.3f) [%s] [%s] [%s] [%s]" % (scores.mean(), 
                                               scores.std(), 
                                               label, 
                                               la, 
                                               al, 
                                               it))


def select_params_for_LinearDiscriminantAnalysis(df_res, y):
    estimators_list = ['lsqr', 'eigen']
    label = 'LinearDiscriminantAnalysis'
    for estim in estimators_list:
        clf = LinearDiscriminantAnalysis(solver=estim, shrinkage=None)
        scores = cross_val_score(clf, df_res, y, cv=10, n_jobs=-1, scoring='roc_auc')
        print("ROC_AUC scoring: %0.3f (+/- %0.3f) [%s] [%s]" % (scores.mean(), 
                                               scores.std(), 
                                               label, 
                                               estim))
        

def select_params_for_SVC(df_res, y):
    C_list = [0.001, 0.01, 0.1, 1, 2, 3, 4]
    kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
    label = 'SVC'
    for c in C_list:
        for k in kernel_list:
            clf = SVC(C=c, kernel=k)
            scores = cross_val_score(clf, df_res, y, cv=10, n_jobs=-1, scoring='roc_auc')
            print("ROC_AUC scoring: %0.3f (+/- %0.3f) [%s] [%s] [%s]" % (scores.mean(), 
                                                   scores.std(), 
                                                   label, 
                                                   c,
                                                   k))
        

if __name__ == '__main__':

    final_list = main(data_path)
    final_list_cleaned = final_list_clean(final_list)
    total_dataframe = list_to_frame(final_list_cleaned)

    y = total_dataframe['Target']
    X = total_dataframe.drop('Target', 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42,
                                                        shuffle=True)
#    classifiers_evaluation(X, y)
#    short_soft_voting(X_train, y_train)

