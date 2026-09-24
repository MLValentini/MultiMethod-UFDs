import os
import numpy as np
import pandas as pd
from utils.util import *
from models import *
import time
from analysis import *
#from sklearn.metrics import calinski_harabasz_score

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

import json

if __name__ == '__main__':

    data = pd.read_csv('data/BootesI_rr_final.csv', header=0)

    data_path = "data/vari_class_18mag.csv"

    is_plot = True
    is_feature_selection = True
    datasets, results = list(), list()

    #fill_csv(data_path) # fill 'Nan' spaces

    initial_data = pd.read_csv(data_path)
    #print(initial_data)

    # read dataset, check columns and select datasets for clustering
    rrl_dataset, mapping_id = read_csv_dataset(data_path)
    
    #check_columns(rrl_dataset)
    datasets, title_list, data_id = select_datasets(rrl_dataset)


    # Check of number of columns
    for idx, ds in enumerate(datasets):
        num_cols = ds.shape[1] if hasattr(ds, 'shape') else len(ds[0])
        print(f"Dataset has {num_cols} columns")


    data_id = np.array([int(i) for i in data_id])

    # scaler
    tmp_datasets = datasets.copy() # orginal dataset (copy) before scaler
    datasets.clear()
    
    # scalers = standard, robust, minmax, abs, normalizer, quantile, power
    [datasets.append(apply_scaler(dataset, "standard")) for dataset in tmp_datasets]


    ### Research of source_ids of target sources (25 RRLs in Bootes I)

    target_ids = data['source_id'].values
    target_indices = initial_data[initial_data['source_id'].isin(target_ids)].index.values
    print(f"{len(target_indices)} target sources found for optimisation")


    # Apply clustering algorithms 
    for title, dataset in enumerate(datasets):
        start = time.time()

        ### Extraction of features from dataset
        _, datas = split_data(dataset)
        
        ### Research of best hyperparameters for DBSCAN and HDBSCAN
        best_db_params = optimize_clustering_for_target(datas, target_indices, model_type='dbscan')
        best_hdb_params = optimize_clustering_for_target(datas, target_indices, model_type='hdbscan')

        ### Run of the models with the found hyperparameters
        results.append(apply_dbscan(dataset, tmp_datasets[title], title_list[title], mapping_id, **best_db_params))
        results.append(apply_hdbscan(dataset, tmp_datasets[title], title_list[title], mapping_id, **best_hdb_params))
        
        end = time.time()
        print("Computation time: " + str(end - start) + " s")

        #if is_plot: plot_clusters(results)
        
        labels_dbscan = results[0][0].labels_
        labels_hdbscan = results[1][0].labels_

        # Feature analysis
        if is_feature_selection:
            # Note: 'dataset' contains already scaled data used into models
            analyze_cluster_features(dataset, labels_dbscan, "dbscan", title_list[title])
            analyze_cluster_features(dataset, labels_hdbscan, "hdbscan", title_list[title])
        
        
        id_labels_dbscan = pd.DataFrame({'source_id': initial_data['source_id'], 'label': labels_dbscan})
        id_labels_hdbscan = pd.DataFrame({'source_id': initial_data['source_id'], 'label': labels_hdbscan})


        ### File Saving

        with open('output/eps_ms_dbscan.dat', 'w') as file:
            json.dump(best_db_params, file)

        with open('output/mcs_ms_hdbscan.dat', 'w') as file:
            json.dump(best_hdb_params, file)

        id_labels_dbscan.to_csv('output/id_labels_dbscan.csv', index=False)
        id_labels_hdbscan.to_csv('output/id_labels_hdbscan.csv', index=False)
        
        results.clear()
        