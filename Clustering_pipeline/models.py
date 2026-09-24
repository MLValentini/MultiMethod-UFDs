"""
Algorithm used:
    - K-means: Classico algoritmo di clustering che, se impostato per creare esattamente due cluster (k=2), suddivide i dati in 
    due gruppi in base alla minimizzazione della somma delle distanze quadrate tra i punti e il centroide del cluster.
    (ENG: Classic clustering algorithm that, when configured to create exactly two clusters (k=2), splits the data 
    into two groups based on minimizing the sum of squared distances between the points and the cluster centroid.)
    
    - Mean Shift: Algoritmo di clustering che cerca le modalità (peaks) nella distribuzione di densità di punti dati. 
    Può essere impostato per trovare due cluster, anche se normalmente è usato per un numero variabile di cluster.
    (ENG: Clustering algorithm that looks for modes (peaks) in the data point density distribution. It can be set to 
    find two clusters, although it is normally used for a variable number of clusters.)
    
    - Spectral Clustering: Utilizza lo spettro (autovalori) della matrice di affinità dei dati per eseguire il clustering. 
    Può essere configurato per due cluster specifici.
    (ENG: Uses the spectrum (eigenvalues) of the data affinity matrix to perform clustering. It can be configured 
    for two specific clusters.)
    
    - Agglomerative Clustering (Hierarchical Clustering): Un algoritmo gerarchico che inizia con ogni punto come un singolo 
    cluster e successivamente fonde i cluster fino a ottenere il numero desiderato (in questo caso, due).
    (ENG: A hierarchical algorithm that starts with each point as a single cluster and successively merges clusters 
    until the desired number (in this case, two) is reached.)
    
    - HDBSCAN (Hiearchical Density-Based Spatial Clustering of Applications with Noise): Algoritmo basato sulla densità che identifica 
    aree ad alta densità come cluster. Anche se non è progettato specificamente per due cluster, può essere manipolato 
    tramite i suoi parametri (epsilon e minPts) per ottenere due cluster.
    (ENG: A density-based algorithm that identifies high-density areas as clusters. Although not specifically 
    designed for two clusters, it can be manipulated through its parameters (epsilon and minPts) to obtain two clusters.)
"""

import matplotlib.pyplot as plt
import numpy as np
from utils.util import map_label_to_points, split_data
from sklearn.cluster import (DBSCAN, OPTICS, AffinityPropagation, 
                             MeanShift, KMeans, SpectralClustering, 
                             AgglomerativeClustering)

from sklearn_extra.cluster import CLARA, KMedoids

import hdbscan
from scipy.cluster.hierarchy import fcluster
from denclue import DENCLUE

def apply_kmeans(data, tmp_dataset, title, mapping_id):
    print("K-means starting...")

    _, datas = split_data(data)
    clusterer = KMeans(n_clusters=2, n_init='auto', random_state=42)
    clusterer.fit(datas)
    # map_label_to_points(clusterer, data, mapping_id, "K-means Clustering " + title)

    return [clusterer, tmp_dataset, "K-means Clustering " + title, data.shape[1]]

def apply_pam(data, tmp_dataset, title, mapping_id):
    print("PAM starting...")

    _, datas = split_data(data)
    clusterer = KMedoids(n_clusters=2, method='pam', random_state=42)
    clusterer.fit(datas)
    # map_label_to_points(clusterer, data, mapping_id, "PAM Clustering " + title)

    return [clusterer, tmp_dataset, "PAM Clustering " + title, data.shape[1]]

def apply_spectralclustering(data, tmp_dataset, title, mapping_id):
    print("Spectral clustering starting...")

    _, datas = split_data(data)
    clusterer = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42, assign_labels='discretize' )
    clusterer.fit_predict(datas)
    # map_label_to_points(clusterer, data, mapping_id, "Spectral Clustering " + title)


    return [clusterer, tmp_dataset, "Spectral Clustering " + title, data.shape[1]]

def apply_agglomerativeclustering(data, tmp_dataset, title, mapping_id):
    print("Agglomerative clustering starting...")

    _, datas = split_data(data)
    clusterer = AgglomerativeClustering(n_clusters=2)
    clusterer.fit_predict(datas)
    # map_label_to_points(clusterer, data, mapping_id, "Agglomerative Clustering " + title)

    return [clusterer, tmp_dataset, "Agglomerative Clustering " + title, data.shape[1]]

def apply_flathdbscan(data, title, mapping_id):
    print("Flat HDBSCAN clustering starting...")

    _, datas = split_data(data)
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(datas)
    Z = clusterer.single_linkage_tree_.to_numpy()
    labels = fcluster(Z, 2, criterion='maxclust')
    # map_label_to_points(labels, data, mapping_id, title)

    return [labels, data, "Flat HDBSCAN Clustering " + title, data.shape[1]]

def apply_meanshift(data, title):
    print("MeanShift starting...")

    clusterer = MeanShift(bandwidth=2)
    clusterer.fit(data)

    return [clusterer, data, "MeanShift Clustering " + title, data.shape[1]]


### 22-08-2026: This will find the best hyperparameters

def apply_dbscan(data, tmp_dataset, title, mapping_id, eps=0.1, min_samples=2):
    """
    DBSCAN has a worst case memory complexity O(n^2), which for 180000 
    samples corresponds to a little more than 259GB.
    This worst case situation can happen if eps is too large or min_samples
    too low, ending with all points being in a same cluster.
    """
    print(f"DBSCAN starting (eps={eps}, min_samples={min_samples})...")
    _, datas = split_data(data)
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    clusterer.fit(datas)
    return [clusterer, tmp_dataset, "DBSCAN Clustering " + title, data.shape[1]]

def apply_hdbscan(data, tmp_dataset, title, mapping_id, min_cluster_size=4, min_samples=2):
    print(f"HDBSCAN starting (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")
    _, datas = split_data(data)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, core_dist_n_jobs=-1)
    clusterer.fit(datas)
    return [clusterer, tmp_dataset, "HDBSCAN Clustering " + title, data.shape[1]]

###


def apply_optics(data, tmp_dataset, title, mapping_id):
    print("OPTICS starting...")

    _, datas = split_data(data)
    clusterer = OPTICS(min_cluster_size=4, min_samples=4, n_jobs=-1)
    clusterer.fit(datas)
    # map_label_to_points(clusterer, data, mapping_id, "OPTICS Clustering " + title)
    return [clusterer, tmp_dataset, "OPTICS Clustering " + title, data.shape[1]]

def apply_denclue(data, title):
    print("DENCLUE starting...")
    
    clusterer = DENCLUE()
    clusterer.fit(data)
    return [clusterer, data, "DENCLUE Clustering " + title, data.shape[1]]

def apply_affinity(data, title):
    print("Affinity Propagation starting...")

    clusterer = AffinityPropagation()
    clusterer.fit(data)

    return [clusterer, data, "Affinity Propagation Clustering " + title, data.shape[1]]