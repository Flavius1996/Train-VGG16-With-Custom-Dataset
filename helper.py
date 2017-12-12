# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:28:54 2017
@author: Haiit

EXTRACT VGG16 DEEP FEATURES &
HELP CREATE TEST SET
"""
import json
import random
from operator import itemgetter
import math

# Create struct of test set
def create_test_set(file_indices_path, num_of_test):
    # load image paths with labels
    with open(file_indices_path) as f:
        files = json.load(f)

    dic = {}
    
    for f in files:
        if f['label_num'] in dic:
            dic[f['label_num']].append(f)
        else:
            dic[f['label_num']] = []
            dic[f['label_num']].append(f)
    
    
    test_sets = []
    
    for i in range(0, num_of_test):
        train_indices = []
        test_indices = []
        for c in dic:
            total_image_in_class = len(dic[c])
            num_test_image = math.ceil (total_image_in_class * 0.2)
            num_train_image = math.floor (total_image_in_class * 0.8)
            random_list = random.sample(range(0, total_image_in_class), total_image_in_class)
            test_idx = random_list[:int(num_test_image)]
            train_idx = random_list[-int(num_train_image):]
            test =[x['id'] for x in itemgetter(*test_idx)(dic[c])]
            train = [x['id'] for x in itemgetter(*train_idx)(dic[c])]
            train_indices.extend(train)
            test_indices.extend(test)
            
        test_ = itemgetter(*test_indices)(files)
        train_ = itemgetter(*train_indices)(files)
        test_sets.append((i, list(train_), list(test_)))
    
    #print(test_sets)
    #print(dic)
    #return None
           
    return test_sets