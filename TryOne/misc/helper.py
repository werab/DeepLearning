#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def calcCategories(y_train):
    cat_count = [0,0,0]
    for a in y_train:
        cat_count += a
    
    # equal class distribution
    cat_weights = {}
    for idx, cat_class in enumerate(cat_count):
        cat_weights[idx] = round(np.max(cat_count)/cat_class)
        
    return cat_count, cat_weights