import torch
import argparse
import numpy as np
import configparser
import pandas as pd
import _pickle as pickle
from m6a_cnn import M6ANet
from m6a_semi_supervised_cnn import tdc, count_pos_neg, make_one_hot_encoded


def find_window(score, precision_score_table):
    for j in range(len(precision_score_table)-1, 1):
            if score >= precision_score_table[j, 1]:
                if score <= precision_score_table[j+1, 1]:
                    #print(j, precision_score_table[j, 1], precision_score_table[j+1, 1])
                    return j
    return 0

def convert_cnn_score_to_int(precision_score_table, float_scores):
    vfind_window = np.vectorize(find_window, excluded=['precision_score_table'])
    unint_score = vfind_window(score=float_scores, precision_score_table=precision_score_table)
    return unint_score

"""

def convert_cnn_score_to_int(precision_score_table, float_scores):
    uint_score = np.zeros((float_scores.shape))
    for i, score in enumerate(float_scores):
        if i % 100000 == 0:
            print(i)
        for j in range(1, len(precision_score_table)-1, 1):
            if score >= precision_score_table[j, 1]:
                if score <= precision_score_table[j+1, 1]:
                    uint_score[i] = j 
                    #print(uint_score[i], precision_score_table[j, 1], score, precision_score_table[j+1, 1])
                    break
    return uint_score
"""               

def make_ont_predictions_255(best_sup_save_model, 
                         data_npz, 
                         output_obj, 
                         precision_score_tsv, 
                         device="cuda"):
    
    precision_score_table = np.loadtxt(precision_score_tsv, 
                                       delimiter="\t", 
                                       dtype=str)
    
    precision_score_table = np.array(precision_score_table[1:, :], dtype=float)
    print(f"precision_score_table: {precision_score_table[0]}")
    
    # Load the supervised model for transfer learning
    model = M6ANet()
    with open(best_sup_save_model, "rb") as fp:
        model.load_state_dict(pickle.load(fp))
        
    model = model.to(device)
    
    val_data = np.load(data_npz)
    X_val = np.array(val_data['features'], dtype=float)
    print(f"X_val: {X_val.shape}")
    
    dorado_score = X_val[:, 5, 7]
    X_val[:, 4, :] = X_val[:, 4, :]/255.0
    X_val[:, 5, :] = X_val[:, 5, :]/255.0
    y_val = np.array(val_data['labels'], dtype=int)
    read_ids = val_data['read_ids']

    #v_hash = np.vectorize(hash)
    #read_id_hashes = v_hash(read_ids)
    
    read_ids_unique = np.unique(read_ids)
    
    read_idx_dict = dict()
    for i, read in enumerate(read_ids_unique):
        read_idx_dict[read] = i
        
    read_id_hashes = np.zeros((read_ids.shape))
    for i, read in enumerate(read_ids):
        read_id_hashes[i] = read_idx_dict[read]
    
    
    positions = val_data['positions']
    # convert to one hot encoded
    y_val_ohe = make_one_hot_encoded(y_val)

    # convert data to tensors
    X_val = torch.tensor(X_val).float()
    y_val_ohe = torch.tensor(y_val_ohe).float()
    #X_val = X_val.to(device)
    #y_val_ohe = y_val_ohe.to(device)
    
    preds_y = model.predict(X_val, device=device)
    total_len = len(preds_y)
    
    preds_y = preds_y[:, 0].numpy()
    preds_y_uint = convert_cnn_score_to_int(precision_score_table, preds_y)
    
    read_id_hashes = read_id_hashes[0:total_len][:, np.newaxis]
    positions = positions[0:total_len][:, np.newaxis]
    y_val = y_val[0:total_len][:, np.newaxis]
    preds_y_uint = preds_y_uint[0:total_len][:, np.newaxis]
    dorado_score = dorado_score[0:total_len][:, np.newaxis]
    
    print(f"read_ids: {read_id_hashes.shape}")
    print(f"positions: {positions.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"preds_y_uint: {preds_y_uint.shape}")
    print(f"dorado_score: {dorado_score.shape}")
    
    output_arr = np.concatenate((read_id_hashes, positions, y_val, dorado_score, preds_y_uint), axis=1)
    output_arr = np.array(output_arr, dtype=float)
    np.savez(output_obj, preds=output_arr)
    
    with open(f"{output_obj}_dict.pkl", 'wb') as f:
        pickle.dump(read_idx_dict, f)


    

    
best_sup_save_model="../models/m6A_ONT_semi_supervised_cnn_5M_set2.best.torch.pickle"
data_npz="/net/gs/vol4/shared/public/hackathon_2023/Stergachis_lab/data/all_sites_npz/merged_00_100p_20k.npz"
output_obj="../results/merged_00_100p_20k_autocorr_input_5M_set2_0_255.npz"
precision_score_tsv="../results/semi_ONT_score_precision_5M_set2.tsv"
#make_ont_predictions_255(best_sup_save_model, data_npz, output_obj, precision_score_tsv)

best_sup_save_model="../models/m6A_ONT_semi_supervised_cnn_5M_set3.best.torch.pickle"
data_npz="/net/gs/vol4/shared/public/hackathon_2023/Stergachis_lab/data/all_sites_npz/merged_00_100p_20k.npz"
output_obj="../results/merged_00_100p_20k_autocorr_input_5M_set3_0_255.npz"
precision_score_tsv="../results/semi_ONT_score_precision_5M_set3.tsv"
#make_ont_predictions_255(best_sup_save_model, data_npz, output_obj, precision_score_tsv)

best_sup_save_model="../models/m6A_ONT_semi_supervised_cnn_5M_set3_run2.best.torch.pickle"
data_npz="/net/gs/vol4/shared/public/hackathon_2023/Stergachis_lab/data/all_sites_npz/merged_00_100p_20k.npz"
output_obj="../results/merged_00_100p_20k_autocorr_input_5M_set3_run2_0_255.npz"
precision_score_tsv="../results/semi_ONT_score_precision_5M_set3_run2.tsv"
#make_ont_predictions_255(best_sup_save_model, data_npz, output_obj, precision_score_tsv)

best_sup_save_model="../models/m6A_ONT_semi_supervised_cnn_5M_set3_run2.best.torch.pickle"
data_npz="/net/gs/vol4/shared/public/hackathon_2023/Stergachis_lab/data/NAPA_raw/HG002_2_NAPA_00.npz"
output_obj="../results/HG002_2_NAPA_00_autocorr_input_5M_set3_run2_0_255.npz"
precision_score_tsv="../results/semi_ONT_score_precision_5M_set3_run2.tsv"
make_ont_predictions_255(best_sup_save_model, data_npz, output_obj, precision_score_tsv)
