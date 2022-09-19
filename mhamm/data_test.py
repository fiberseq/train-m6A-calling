#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:03:48 2022

@author: morgan
"""

import os
import pickle
# import m6a_calling
# from m6a_calling import 

os.chdir(os.path.expanduser("~/git/m6A-calling"))

import m6a_ml_data


# Test that positive pickle works
positive_pickle_path = 'local/PositiveSMRTmatrix.pkl'

positive = pickle.load(open(positive_pickle_path, "rb"))

pos_feats, pos_labels, pos_others = m6a_ml_data.get_feat_labels_matrix(positive, req_label=1)



