# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:20:45 2018

@author: Kirill
"""


import os
import pandas as pd
import csv
import math


#url_data = 'C:/Users/Кирилл/retnna/Data/csv_data/raw_nr/'
#file_name = 'Search_Andrii2_all_gaze.csv'

url_data = 'C:/Users/Кирилл/retnna/Data/csv_data/notreading/'
file_name = 'pct2_olga.csv' #'faces_anastasia.csv'
data_path = 'C:/Users/Кирилл/retnna/Work_data/notreading/'

#data = url_data + file_name


def read_file(data_path):
    file_number = 0
    file_to_work = list()
    for filename in os.listdir(data_path):
        if filename.startswith("male_"):
            file_number += 1
            file_to_work.append(data_path + filename)
        elif filename.startswith("female_"):
            file_number += 1
            file_to_work.append(data_path + filename)
        else:
            continue
    return(file_to_work, file_number)


def read_data(url_data, file_name):
    f_open = open(url_data + file_name)
    return pd.read_csv(f_open, sep='\t', encoding='cp1251')


def read_data_from_file(url_data, file_name):
    with open(url_data + file_name, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)
        for line in reader:
            yield line


def temp_trans(temp_tank):
    from copy import deepcopy
    temp = deepcopy(temp_tank)
    return temp


def separator(url_data, file_name):
    temp_result_list =list()
    result_list = list()
    generator = read_data_from_file(url_data, file_name)
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
        
        
def session_collapsing(session_list):
    horizontal_distance = one_line_dist(session_list)
    vertical_distance = inter_line_dist(session_list)
    total_time = float(session_list[-1][7])
    total_blinks = math.fabs(float(session_list[0][12]) - float(session_list[-1][12]))
    return [horizontal_distance, vertical_distance, total_time, total_blinks]


def result_data_cooking(res):
    res_list = list()
    for session in res:
        res_list.append(session_collapsing(session))
    return res_list
    
    

if __name__ == '__main__':
    frame = read_data(url_data, file_name)
    res = separator(url_data, file_name)
    final_list = result_data_cooking(res)
    
    
#def total_dist(session_list):
#    temp = 0
#    temp_0 = 0
#    for line in session_list:
#        temp += distance(
#                float(line[1]), 
#                float(line[2]), 
#                float(line[7]), 
#                float(line[8]))
#        if session_list.index(line) == 0:
#            continue
#        elif session_list.index(line) == len(session_list):
#            break
#        else:
#            line_0 = session_list[session_list.index(line) + 1]
#            temp_0 += distance(
#                    float(line[7]), 
#                    float(line[8]), 
#                    float(line_0[1]), 
#                    float(line_0[2]))
#    return temp + temp_0