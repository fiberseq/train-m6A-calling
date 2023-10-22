#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:51:51 2023

@author: morgan hamm
"""

import numpy as np
import pandas as pd
import scipy
from scipy import signal
import argparse

# import seaborn as sns

args = argparse.Namespace(npz_file='/home/morgan/Documents/grad_school/misc_code/hackathon/merged_00_100p_20k_autocorr_input_5M_set2.npz',
                          invert_ml=True, ml_cutoff=0.938, dorado_cutoff=0.95, 
                          n_sites=2000, output_file='/home/morgan/Documents/grad_school/misc_code/hackathon/test_out.npz') 


# =============================================================================
# stack = []
# for i in range(2000):
#     subset =  preds[preds['id_hash'] == pos_ids[i]]
#     if subset.iloc[-1]['pos'] > 820:
#         autocorr = auto_corr(subset, score_col="dorado", cutoff=0.37)
#         if (autocorr is not None):
#             stack.append(autocorr)
# 
# temp = np.stack(stack)
# temp2 = np.nansum(temp, axis=0)/temp.shape[0]
# 
# sns.lineplot(lags, temp2)
# =============================================================================

def auto_corr(subset, w_len = 400, big_w_len = 800, score_col = 'ml', cutoff=0.938):
    w_start = int((subset.iloc[-1]['pos'] / 2) - (big_w_len / 2))
    
    filt = subset[(subset['pos'] >= w_start) & ((subset['pos'] < w_start + big_w_len) & (subset[score_col] >= cutoff) ) ]
    
    if len(filt) == 0:
        return(None)
    
    big_window = np.zeros(big_w_len, dtype=float) 
    for i, row in filt.iterrows():
        big_window[int(row['pos']) - w_start] = 1 /len(filt)
    little_window = big_window[0:w_len]
    
    # scale_factor = 1 / len(filt)
    # scale_factor = 1/sum(little_window)
    # scale_factor = 1/sum(big_window)
    # little_window = little_window * scale_factor
    # big_window = big_window * scale_factor
    if sum(little_window) == 0:
        return(None)
    
    autocorr = signal.correlate(big_window, little_window, "valid")
    # lags = sp.signal.correlation_lags(big_w_len, w_len, "valid")
    
    # norm_fact = np.sum(little_window**2)
    # sns.lineplot(lags, autocorr/norm_fact)
    return(autocorr)


def auto_corr_n(preds, score_col, n_sites, cutoff):
    stack = []
    read_ids = np.unique(preds['id_hash'])
    for i in range(n_sites):
        subset =  preds[preds['id_hash'] == read_ids[i]]
        #print(i, len(subset))
        if subset.iloc[-1]['pos'] > 820:
            autocorr = auto_corr(subset, score_col=score_col, cutoff=cutoff)
            #print(f"autocorr")
            if (autocorr is not None):
                #if autocorr.shape[0] == 401:
                    stack.append(autocorr)
    print("stack: ", len(stack))
    if len(stack) > 0:
        all_out = np.stack(stack)
        return(np.nansum(all_out, axis=0)/float(all_out.shape[0]))
    else:
        all_out = stack
        return all_out
    
    

def main(args):
    data = np.load(args.npz_file)
    preds = data['preds']
    preds = pd.DataFrame(preds, columns=['id_hash', 'pos', 'label', 'dorado', 'ml'])
    
    if args.invert_ml == True:
        preds['ml'] = 1 - preds['ml']
    
    # read_ids = np.unique(preds['id_hash'])
    
    preds_neg = preds[preds['label'] == 0]
    preds_pos = preds[preds['label'] == 1]
    
    # pos_ids = np.unique(preds_pos['id_hash'])
    # neg_ids = np.unique(preds_neg['id_hash'])
    

    print("lab1_ml_data")
    lab1_ml_data = auto_corr_n(preds_pos, score_col='ml', n_sites=args.n_sites, cutoff=args.ml_cutoff)
    
    print("lab0_ml_data")
    lab0_ml_data = auto_corr_n(preds_neg, score_col='ml', n_sites=args.n_sites, cutoff=args.ml_cutoff)
    
    print("lab1_dorado_data")
    lab1_dorado_data = auto_corr_n(preds_pos, score_col='dorado', n_sites=args.n_sites, cutoff=args.dorado_cutoff)
    
    print("lab0_dorado_data")
    lab0_dorado_data = auto_corr_n(preds_neg, score_col='dorado', n_sites=args.n_sites, cutoff=args.dorado_cutoff)
    
    npz_struct = {'lab1_ml_data':lab1_ml_data, 
                  'lab0_ml_data':lab0_ml_data, 
                  'lab1_dorado_data':lab1_dorado_data, 
                  'lab0_dorado_data':lab0_dorado_data,}
    
    np.savez(args.output_file, **npz_struct)
        
# ---------------------
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='parse an ONT BAM file into features surrounding called m6As')
    parser.add_argument('npz_file', help='npz file with ML calls for all As in a set of fibers')
    parser.add_argument('-i', '--invert_ml', type=bool, default=False, 
                        help='set ml score to 1 - ML') 
    parser.add_argument('-m', '--ml_cutoff', type=float, default=0.938, 
                        help='cutoff to use for ML results')
    parser.add_argument('-d', '--dorado_cutoff', type=float, default=0.95, 
                        help='cutoff to use for ML results')
    parser.add_argument('-n', '--n_sites', type=int, default=5000, 
                        help='number of sites or number of fibers to look at')
    parser.add_argument('-o', '--output_file', type=str, default='output.npz', 
                        help='output file name prefix') 
    args = parser.parse_args()
    main(args)