# -*- coding: utf-8 -*-
"""
pre-processing.py
Created on 22-01-2024 

@author: Lorenzo Monti
@email: lorenzo.monti@inaf.it

Pre-processing process in order to prepare the dataset
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # no UI backend
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import seaborn as sns
from pyclustertend import hopkins, vat
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, Normalizer, QuantileTransformer, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


### Added on 22-08-2026

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
from sklearn.cluster import DBSCAN
import hdbscan

### 

### Added on 11-09-2026

from sklearn.utils import resample
import gc

###


def read_csv_dataset(data_path):
    csv_data = pd.read_csv(data_path)
    df = pd.DataFrame(csv_data)

    df['id'] = np.arange(0, len(df) * 1.0, 1.0)
    mapping_id = df[["source_id", "id"]]
    catalogue = df.to_numpy()

    return catalogue, mapping_id

def fill_csv(data_path):
    # load the CSV file into a Pandas dataframe
    df = pd.read_csv(data_path)
    #df.dropna()
    df.fillna(11111111, inplace=True) # fill with 11111111 where there are 'Nan'
    df.to_csv(data_path, index=False)

def plot_dataset(x, y):
    print("plot dataset")
    plt.plot(x, y, 'o') 
    plt.title('Plot')
    plt.xlabel('Period')
    plt.ylabel('Amplitude')
    plt.show()
    plt.close()

def map_label_to_points(clusterer, dataset, mapping_id, title):

    # Get the labels
    labels = clusterer.labels_

    # Map labels to points
    label_to_points = {}
    for label, point in zip(labels, dataset):
        if label not in label_to_points:
            label_to_points[label] = []
        label_to_points[label].append(point[-1])

    # Convert the dictionary values to numpy arrays for better readability
    for label in label_to_points:
        label_to_points[label] = np.array(label_to_points[label])

    export_mapped_data(label_to_points, mapping_id, f"output/{title}.csv")

def export_mapped_data(label_to_points, mapping_id, output_csv_path):
    # Convert mapping_id to a DataFrame for easier manipulation
    mapping_df = pd.DataFrame(mapping_id, columns=['source_id', 'id'])
    mapping_df['source_id'] = mapping_df['source_id'].astype(str)
    
    # Initialize a list to hold the results
    results = []

    # Iterate over each key in the label_to_points dictionary
    for key, points_array in label_to_points.items():
        if key == -1:
            continue
        # Get the IDs from the mapping array that are present in the current points_array
        matching_mapping = mapping_df[mapping_df['id'].isin(points_array)]

        # Append the results with the key and id_source
        for _, row in matching_mapping.iterrows():
            results.append([key, row['source_id']])
    
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results, columns=['label', 'source_id'])

    # Export the DataFrame to a CSV file
    results_df.to_csv(output_csv_path, index=False)

    print(f"Data has been exported to {output_csv_path}")

def split_data(data):
    id_source = data[:, -1] # id
    data = data[:, :-1] # remove id from nd array
    return id_source, data

def check_columns(dataset):
    """
    Check if all columns are full filled
    """
    check_list = []
    [check_list.append(dataset[:, 1].shape[0]) for i in range(dataset.shape[1])]
    for element in check_list:
        if check_list[0] != element:
            print("Some element in dataset are missing")
            break
    print("Dataset checked.")

def select_features(dataset, features):
    return dataset[:,features]

def cluster_tendency(dataset):
    print("Hopkins cluster tendency")
    print(hopkins(scale(dataset), dataset.shape[0]))
    #print("VAT cluster tendency")
    #vat(scale(dataset))
    #print("iVAT cluster tendency")
    #ivat(scale(dataset))

def select_datasets(rrl_dataset):

    # header of 'vari_class_18mag.csv': 
    #   "source_id", "ra", "dec", "pmra", "pmra_error", "pmdec", "pmdec_error", 
    #   "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "w_mag"

    datasets = list()
    title_list = ["source_id", "ra", "dec", "pmra", "pmdec", "phot_g_mean_mag", "w_mag"] # our interest
    

    # p1peaktopeakr21phi21
    data = select_features(rrl_dataset, [0, 1, 2, 3, 5, 7, 10])
    #print(data)
    data_s, data_id = preprocess_dataset(data)
    datasets.append(data_s)

    #print(data_s)

    #print(len(datasets[0]), len(datasets[0][0]))

    return datasets, title_list, data_id

def preprocess_dataset(data): # remove datas with 'Nan' == 11111111.0 (already cleaned so we comment part of this)

    df_dataset = pd.DataFrame(data) # convert in dataframe

    #print(df_dataset)

    #df_dataset = df_dataset[df_dataset.iloc[: , 3] != 11111111.0] # drop if pmra = 11111111 (Nan)
    #df_dataset = df_dataset[df_dataset.iloc[: , 4] != 11111111.0] # drop if pmdec = 11111111 (Nan)
    #df_dataset = df_dataset[df_dataset.iloc[: , 5] >= 18.0] # drop if phot_g_mean_mag < 18 mag
    #df_dataset = df_dataset.drop(columns= [5], axis = 1) # tolgo le colonne di "phot_g_mean_mag"
    #df_dataset = df_dataset.drop(columns= [5,6,7], axis = 1) # tolgo le colonne di "phot_g_mean_mag","phot_bp_mean_mag","phot_rp_mean_mag"

    #print(df_dataset)

    df_sourceid = df_dataset.iloc[: , 0] # list of source_id
    
    #print(df_sourceid)

    df_dataset = df_dataset.iloc[: , 1:] # dataset slpitted from source_id

    #print(df_dataset)

    #print(df_sourceid)
    #print(df_dataset.shape)

    return df_dataset.to_numpy(), df_sourceid.to_numpy()

def apply_scaler(X, scaler):
    id_source, X = split_data(X)
    X = choose_scaler(scaler).fit_transform(X)
    data = np.concatenate((X, id_source.reshape(-1, 1)), axis=1)
    return data

def choose_scaler(scaler): # this applies a scale on data (Normalisation)
    if scaler == "standard":
        scaler = StandardScaler()
    elif "robust":
        scaler = RobustScaler()
    elif "minmax":
        scaler = MinMaxScaler()
    elif "abs":
        scaler = MaxAbsScaler()
    elif "normalizer":
        scaler = Normalizer()
    elif "quantile":
        scaler = QuantileTransformer()
    elif "power":
        scaler = PowerTransformer()
    else:
        scaler = StandardScaler()
    
    return scaler

def plot_clusters(results):

    # result = [clusterer, data, title, dimension] 
    for result in results:

        if result[3] > 2: # dataset dimension (3d+)
            data = result[1][:,:2]
        else: data = result[1]
        plot_2d(result[0], data, result[2])

def plot_2d(clusterer, data, title):
    color_palette = sns.color_palette('colorblind', len(clusterer.labels_))
    cluster_colors = [color_palette[x] if x >= 0
                    else (0.5, 0.5, 0.5)
                    for x in clusterer.labels_]
    plt.scatter(*data.T, s=.5, linewidth=0, c=cluster_colors, alpha=0.85)
    plt.title(title)
    plt.xlabel("ra")
    plt.ylabel("dec")
    #plt.show()
    #plt.savefig("plot/" + title + '.png', format='png', dpi=1200)
    plt.close()

def plot_pm(clusterer, data, title):
    color_palette = sns.color_palette('colorblind', len(clusterer.labels_))
    cluster_colors = [color_palette[x] if x >= 0
                    else (0.5, 0.5, 0.5)
                    for x in clusterer.labels_]
    plt.scatter(*data.T, s=.5, linewidth=0, c=cluster_colors, alpha=0.85)
    plt.title(title)
    plt.xlabel("pmra")
    plt.ylabel("pmdec")
    #plt.show()
    #plt.savefig("plot/" + title + ' pm.png', format='png', dpi=1200)
    plt.close()

def preprocess_astronomical_data(df):
    """
    Preprocess astronomical features specific to this dataset
    
    Parameters:
    df (pandas.DataFrame): Input dataframe with astronomical features
    
    Returns:
    pandas.DataFrame: Preprocessed dataframe
    """
    processed_df = df.copy()
    
    # get log10(pf)
    #processed_df['pf'] = np.log10(processed_df['pf'])

    """    # Handle magnitude-based features
    mag_features = ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'grvs_mag']
    
    # Create color indices if not already present
    if 'bp_rp' not in processed_df.columns:
        processed_df['bp_rp'] = processed_df['phot_bp_mean_mag'] - processed_df['phot_rp_mean_mag']
    if 'bp_g' not in processed_df.columns:
        processed_df['bp_g'] = processed_df['phot_bp_mean_mag'] - processed_df['phot_g_mean_mag']
    if 'g_rp' not in processed_df.columns:
        processed_df['g_rp'] = processed_df['phot_g_mean_mag'] - processed_df['phot_rp_mean_mag']
    
    # Handle phase-based features
    phase_features = ['phi21_g', 'phi31_g']
    processed_df[phase_features] = processed_df[phase_features].apply(lambda x: np.sin(np.radians(x)))
    
    # Create proper motion magnitude
    if 'pmra' in processed_df.columns and 'pmdec' in processed_df.columns:
        processed_df['pm_total'] = np.sqrt(processed_df['pmra']**2 + processed_df['pmdec']**2)"""
    
    # Handle missing values
    processed_df = processed_df.fillna(processed_df.median())
    
    return processed_df

def get_scaler(X):
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)

def perform_feature_selection(df, target_column, n_features=10):
    """
    Perform feature selection on astronomical data for regression
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    target_column (str): Name of the target variable column
    n_features (int): Number of top features to select
    
    Returns:
    dict: Dictionary containing selected features from different methods
    """
    # Preprocess the data
    df_processed = preprocess_astronomical_data(df)
   
    # Separate features and target
    X = df_processed.drop(target_column, axis=1)
    y = df_processed[target_column]
    
    # Scale the features
    X_scaled = get_scaler(X)

    # Dictionary to store results
    selected_features = {}
    
    # 1. Pearson Correlation
    correlations = pd.DataFrame({
        'feature': X.columns,
        'correlation': [abs(X_scaled[col].corr(y, method='pearson')) for col in X.columns]
    })
    correlations = correlations.sort_values('correlation', ascending=False)
    selected_features['correlation'] = correlations.head(n_features)['feature'].tolist()
    
    # 2. ANOVA F-value (for regression)
    selector_f = SelectKBest(score_func=f_regression, k=n_features)
    selector_f.fit(X_scaled, y)
    f_scores = pd.DataFrame({
        'feature': X.columns,
        'f_score': selector_f.scores_
    })
    selected_features['f_score'] = f_scores.nlargest(n_features, 'f_score')['feature'].tolist()
    
    # 3. Mutual Information (for regression)
    selector_mi = SelectKBest(score_func=mutual_info_regression, k=n_features)
    selector_mi.fit(X_scaled, y)
    mi_scores = pd.DataFrame({
        'feature': X.columns,
        'mi_score': selector_mi.scores_
    })
    selected_features['mutual_info'] = mi_scores.nlargest(n_features, 'mi_score')['feature'].tolist()
    
    # 4. Random Forest Feature Importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    importance_scores = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    })
    selected_features['random_forest'] = importance_scores.nlargest(n_features, 'importance')['feature'].tolist()
    print(selected_features['random_forest'] )
    return df_processed, selected_features, correlations, f_scores, mi_scores, importance_scores

def compute_regression_metrics(y, y_pred, metrics: dict = None, sample_weight=None):
    """
    Compute regression metrics and append them to list of metrics in a dictionary.
    :param y: numpy.ndarray
        Array of the true values.
    :param y_pred: numpy.ndarray
        Array of the predicted values.
    :param metrics: dict or None
        A dictionary of metric lists. If None, a new dictionary will be created.
    :param sample_weight: numpy.ndarray or None
        Array of the sample weights.
    :return: metrics: dict
        Dictionary of metric lists.
    """    
    metrics = {'r2': [], 'wrmse': [], 'wmae': [], 'rmse': [], 'mae': []}

    if sample_weight is not None:
        metrics['r2'].append(r2_score(y, y_pred, sample_weight=sample_weight))
        metrics['wrmse'].append(np.sqrt(mean_squared_error(y, y_pred, sample_weight=sample_weight)))
        metrics['wmae'].append(mean_absolute_error(y, y_pred, sample_weight=sample_weight))
    metrics['r2'].append(r2_score(y, y_pred))
    metrics['rmse'].append(np.sqrt(mean_squared_error(y, y_pred)))
    metrics['mae'].append(mean_absolute_error(y, y_pred))

    return metrics

def visualize_feature_importance(selected_features, all_scores):
    """
    Create visualizations for feature importance scores
    """
    correlations, f_scores, mi_scores, importance_scores = all_scores

    # Create a summary of feature selection frequency
    feature_counts = {}
    for method in selected_features.values():
        for feature in method:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    summary_df = pd.DataFrame.from_dict(feature_counts, orient='index', columns=['count'])
    summary_df = summary_df.sort_values('count', ascending=True)
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot 1: Feature Selection Frequency
    summary_df.plot(kind='barh', ax=axes[0])
    axes[0].set_title('Feature Selection Frequency Across Methods')
    axes[0].set_xlabel('Number of Methods Selected')
    axes[0].set_ylabel('Features')
    
    # Plot 2: Random Forest Feature Importance
    importance_scores = importance_scores.sort_values('importance', ascending=True)
    importance_scores.plot(kind='barh', x='feature', y='importance', ax=axes[1])
    axes[1].set_title('Random Forest Feature Importance')
    axes[1].set_xlabel('Importance Score')
    axes[1].set_ylabel('Features')
    
    plt.tight_layout()
    return fig


### 22-08-2026: This will find the best hyperparameters

def _target_cluster_scorer(labels, target_indices):

    # This computes the maximum F1-score obtained by a cluster on target sources
    unique_labels = set(labels) - {-1}
    if not unique_labels:
        return 0.0
    
    y_true = np.zeros(len(labels), dtype=bool)
    y_true[target_indices] = True
    
    best_f1 = 0.0
    for label in unique_labels:
        y_pred = (labels == label)
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
    return best_f1

def optimize_clustering_for_target(datas, target_indices, model_type='dbscan'):
    
    # This find the combination of hyperparameters which maximise F1-score for the targer sources
    print(f"Optimizing parameters for {model_type.upper()}...")
    
    if model_type == 'dbscan':
        param_grid = {
            #'eps': np.linspace(0.01, 0.1, 10),  # Grid for data already scaled
            #'min_samples': [2, 3, 4, 5, 6]
            'eps': [0.1],  # Grid for data already scaled
            'min_samples': [2]
        }
    elif model_type == 'hdbscan':
        param_grid = {
            #'min_cluster_size': [2, 3, 4, 5, 6],
            #'min_samples': [2, 3, 4, 5, 6]
            'min_cluster_size': [4],
            'min_samples': [2]
        }
    
    best_score = -1.0
    best_params = {}
    
    for params in ParameterGrid(param_grid):
        if model_type == 'dbscan':
            clusterer = DBSCAN(**params, n_jobs=-1)
        else:
            clusterer = hdbscan.HDBSCAN(**params, core_dist_n_jobs=-1)
            
        clusterer.fit(datas)
        score = _target_cluster_scorer(clusterer.labels_, target_indices)
        
        if score > best_score:
            best_score = score
            best_params = params
            
    print(f"[{model_type.upper()}] Best F1-Score: {best_score:.4f} with params: {best_params}")
    return best_params

###


### 11-09-2026: This handles noise point filtering (-1), Random Forest training, and saving feature weights to a CSV file.

def analyze_cluster_features(dataset, labels, algo_name, title):
    
    ## Analysis and saving the feature "importance" for the clusters performed by DBSCAN and HDBSCAN

    # Convertion in DataFrame Pandas if it is an NumPy array
    if isinstance(dataset, np.ndarray):
        df_features = pd.DataFrame(dataset)
    else:
        df_features = dataset.copy()
        
    # Mask to exclude the noise (label = -1)
    mask = labels != -1
    #X_clean = df_features[mask]
    #labels_clean = labels[mask]

    if isinstance(dataset, np.ndarray):
        X_clean = pd.DataFrame(dataset)[mask]
    else:
        X_clean = dataset[mask].copy()
        
    labels_clean = labels[mask]
    
    # If there are 0 or 1 valid cluster, we cannot train the classifier
    unique_labels = np.unique(labels_clean)
    if len(unique_labels) < 2:
        print(f"[{algo_name} - {title}] Too few clusters for analysis (found only {unique_labels}). Skipping.")
        return None

    # SUB-SAMPLING: This limits to max. 20.000 points to prevent memory overload (RAM)
    if len(X_clean) > 20000:
        X_clean, labels_clean = resample(
            X_clean, labels_clean, 
            n_samples=20000, 
            stratify=labels_clean, 
            random_state=42
        )

    # Random Forest Classifier training
    #rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    #rf.fit(X_clean, labels_clean)

    # Random Forest with depth limits and reduced threads
    rf = RandomForestClassifier(
        n_estimators=50,      # Reduced from 100 to 50
        max_depth=12,         # Prevents trees from taking up too much RAM
        n_jobs=2,             # Prevents core and memory saturation
        random_state=42
    )
    rf.fit(X_clean, labels_clean)

    # Generate DataFrame of "importance"
    #feature_names = df_features.columns if hasattr(df_features, 'columns') else [f"feature_{i}" for i in range(df_features.shape[1])]
    
    #importance_df = pd.DataFrame({
    #    'feature': feature_names,
    #    'importance': rf.feature_importances_
    #}).sort_values('importance', ascending=False)

    # Extraction of names and saving
    # If 'dataset' is a DataFrame Pandas, this takes its column names
    if isinstance(dataset, pd.DataFrame):
        feature_names = dataset.columns.tolist()
    # If 'dataset' is a NumPy array, but we give a 'support' DataFrame or a list of names
    #elif hasattr(dataset, 'columns'):
    #    feature_names = list(dataset.columns)
    else:
        # Fallback if there are no available names
        feature_names = ["ra", "dec", "pmra", "pmdec", "phot_g_mean_mag", "w_mag"]

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Results saving in .csv
    output_filename = f"output/feature_importance_{algo_name}.csv"
    importance_df.to_csv(output_filename, index=False)
    #print(f"[{algo_name} - {title}] Feature importance saved in: {output_filename}")

    print(f"\n TOP 6 important Features for [{algo_name.upper()} - {title}]:")
    print(importance_df.head(11).to_string(index=False))

    # Explicit cleaning of RAM
    del X_clean, labels_clean, rf
    gc.collect()

    return importance_df

###