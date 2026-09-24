import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg') # no UI backend
import matplotlib.pyplot as plt
import seaborn as sns

def plot_2d(clusterer, data, title):
    color_palette = sns.color_palette('colorblind', len(clusterer.labels_))
    cluster_colors = [color_palette[x] if x >= 0
                    else (0.5, 0.5, 0.5)
                    for x in clusterer.labels_]
    plt.scatter(*data.T, s=.5, linewidth=0, c=cluster_colors, alpha=0.85)
    plt.title(title)
    plt.xlabel("pm_ra")
    plt.ylabel("pm_dec")
    #plt.show()
    #plt.savefig("plot/" + title + '.png', format='png', dpi=1200)
    plt.close()