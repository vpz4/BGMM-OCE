# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 09:58:29 2021

@author: bpez
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import matplotlib
import warnings
from scipy import stats
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from minisom import MiniSom


matplotlib.use('Agg')
warnings.filterwarnings("ignore")
os.chdir(os.getcwd())
plt.rcParams.update({'font.size': 7, 'figure.figsize': (26, 22)})


def plot_db_scores(cluster_counts, db_scores, optimal_clusters=None):
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_counts, db_scores, marker='o', linestyle='-', color='b')
    
    if optimal_clusters is not None:
        optimal_index = cluster_counts.index(optimal_clusters)
        optimal_score = db_scores[optimal_index]
        plt.scatter(optimal_clusters, optimal_score, color='red', s=100, label=f'Optimal ({optimal_clusters} clusters)')
        plt.legend()
    
    plt.title('Davies-Bouldin Scores by Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Bouldin Score')
    plt.grid(True)
    plt.xticks(cluster_counts)
    plt.show()
    plt.savefig('synthetic/DBscores.png', dpi=300)


def perform_clustering(data, num_clusters):
    som_dim = int(np.sqrt(num_clusters))
    sigma_value = min(som_dim / 2, 1.0)
    som = MiniSom(som_dim, som_dim, data.shape[1], sigma=sigma_value, learning_rate=0.5)
    som.random_weights_init(data.values)
    som.train_random(data.values, 500)

    # Use train_batch for deterministic training
    # som.train_batch(data.values, num_iteration=500)

    labels = np.array([som.winner(d)[0] * som_dim + som.winner(d)[1] for d in data.values])
    cluster_map = {}
    for idx, label in enumerate(labels):
        if label not in cluster_map:
            cluster_map[label] = []
        cluster_map[label].append(idx)

    return labels, cluster_map, som


def find_optimal_clusters(data, max_clusters):
    db_scores = []
    cluster_counts = []
    for clusters in range(2, max_clusters + 1):
        labels, cluster_map, _ = perform_clustering(data, clusters)
        unique_labels = np.unique(labels)
        
        if len(unique_labels) > 1:
            score = davies_bouldin_score(data.values, labels)
            db_scores.append(score)
            cluster_counts.append(clusters)
            print(f"DB Score for {clusters} clusters: {score}")
        else:
            print(f"Insufficient clusters ({len(unique_labels)}) for DB score calculation at {clusters} clusters.")
    
    min_score_index = db_scores.index(min(db_scores))
    optimal_clusters = cluster_counts[min_score_index]

    return cluster_counts, db_scores, optimal_clusters


def collect_categorical_mappings(dataframe, threshold=10):
    categorical_mappings = {}
    for column in dataframe.columns:
        unique_values = dataframe[column].dropna().unique()
        if len(unique_values) <= threshold:
            categorical_mappings[column] = sorted(unique_values)  # Sort to ensure consistent mapping

    return categorical_mappings


def map_to_category(values, categories):
    categories = np.array(categories)
    category_indices = np.abs(values[:, None] - categories[None, :]).argmin(axis=1)

    return categories[category_indices]


D = pd.read_csv('test.csv')
D.drop(columns=['Idiopathic Inflammatory Myopathy (IIM)',
                'Inclusion Body Myositis (IBM) documented with Biopsy'], inplace=True)
feature_names = list(D.columns)
D = pd.DataFrame(columns=feature_names, data=D)
categorical_mappings = collect_categorical_mappings(D)

X_corr = D.corr(method='pearson')
X_cov = D.cov()
# cV_real = D.std()/D.mean()

plt.figure()
sns.heatmap(D.corr())
plt.savefig('synthetic/Corr_plot_real.png', dpi=600, bbox_inches='tight')
plt.close()

num_patients = 2000
max_clusters = 30

cluster_counts, db_scores, optimal_clusters = find_optimal_clusters(D, max_clusters)

print(f"The optimal number of clusters is {optimal_clusters}")
plot_db_scores(cluster_counts, db_scores, optimal_clusters)

weight_concentration_prior = np.exp(-optimal_clusters)

# fit BGMM
start = time.time()
g = BayesianGaussianMixture(n_components=optimal_clusters,
                            covariance_type='diag',
                            random_state=0,
                            weight_concentration_prior=weight_concentration_prior)
g.fit(D)
end = time.time()
print(f"Elapsed time {end-start}")

gofss = []
KL_divss = []
scoress = []
cof_dif_scoress = []
cov_dif_scoress = []
std_cof_dif_scoress = []
std_cov_dif_scoress = []
weights_opts = []
opts = []
times = []
CORR = []
COV = []
cV_diff = []
cV_virtual = []
vmrrr = []
        
path_root = 'synthetic/'+str(num_patients)

if(not os.path.isdir(path_root)):
    os.makedirs(path_root)
print("Number of virtual patients =", num_patients)

start = time.time()
Xnew = g.sample(num_patients)[0]
X_VP = pd.DataFrame(columns=D.columns, data=Xnew)

for col, categories in categorical_mappings.items():
    if col in X_VP.columns:
        X_VP[col] = map_to_category(X_VP[col].values, categories)

X_VP.to_csv(path_root+'/Vpop_'+str(num_patients)+'.csv', index=False)

end = time.time()
times.append(end-start)
print(f"Elapsed time = {end-start} sec")
print("")

aa = []
p_aa = []
KL_divv = []
vmrr = []
i = 0
for s in list(D.columns):
    [a, p_a] = stats.ks_2samp(D.iloc[:,i], X_VP.iloc[:,i])
    aa.append(a)
    p_aa.append(p_a)
    kd_r = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(D.iloc[:,i].values.reshape(-1, 1))
    kd_v = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X_VP.iloc[:,i].values.reshape(-1, 1))
    s = np.linspace(0,200)
    er = kd_r.score_samples(s.reshape(-1, 1))
    ev = kd_v.score_samples(s.reshape(-1, 1))
    KL_div = stats.entropy(er, ev)
    
    if(len(np.unique(X_VP.iloc[:,i])) == 2):
        vmr_r = np.std(D.iloc[:,i])/np.median(D.iloc[:,i])
        vmr_v = np.std(X_VP.iloc[:,i])/np.median(X_VP.iloc[:,i])
    elif(len(np.unique(X_VP.iloc[:,i])) == 1):
        vmr_r = (np.std(D.iloc[:,i])+0.000001)/(np.median(D.iloc[:,i])+0.000001)
        vmr_v = (np.std(X_VP.iloc[:,i])+0.000001)/(np.median(X_VP.iloc[:,i])+0.000001)
    else:
        vmr_r = np.std(D.iloc[:,i])/np.mean(D.iloc[:,i])
        vmr_v = np.std(X_VP.iloc[:,i])/np.mean(X_VP.iloc[:,i])  
    vmr_diff = np.abs(vmr_r-vmr_v)
    KL_divv.append(KL_div)
    vmrr.append(vmr_diff)
    i += 1
    
X_VP_corr = X_VP.corr(method='pearson')
X_VP_corr.fillna(1, inplace=True)
X_diff = np.abs(X_corr-X_VP_corr)
diff_metric = X_diff.values[np.triu_indices_from(X_diff.values,1)].mean()
std_metric  = X_diff.values[np.triu_indices_from(X_diff.values,1)].std()
CORR.append(X_diff)

X_VP_cov = X_VP.cov()
X_diff2 = np.abs(X_cov-X_VP_cov)
diff_metric2 = X_diff2.values[np.triu_indices_from(X_diff2.values,1)].mean()
std_metric2  = X_diff2.values[np.triu_indices_from(X_diff2.values,1)].std()
COV.append(X_diff2)

cV_virtual.append(X_VP.std()/X_VP.mean())
cV_diff.append(np.abs((D.std()/D.mean())-(X_VP.std()/X_VP.mean())))

# gofss.append(np.mean(aa))
# KL_divss.append(np.mean(KL_divv))
# scoress.append(g.score(D))
# cof_dif_scoress.append(diff_metric)
# cov_dif_scoress.append(diff_metric2)
# std_cof_dif_scoress.append(std_metric)
# std_cov_dif_scoress.append(std_metric2)
# weights_opts.append(weight_concentration_prior)

print(f"Average gof = {np.mean(aa)}")
print(f"Average KL-divergence = {np.mean(KL_divv)}")
print(f"Average cV difference = {np.mean(cV_diff)}")
print(f"Average CORR difference = {np.mean(diff_metric)}")
print(f"Average COV difference = {np.mean(diff_metric2)}")

results_DF = pd.DataFrame(data={'gof':aa,
                                'KL_div':KL_divv,
                                'cV diff.':np.abs((D.std()/D.mean())-(X_VP.std()/X_VP.mean())),
                                'Correlation diff.': diff_metric,
                                'Covariance diff.': diff_metric2,
                            })
results_DF.to_csv(path_root+'/results.csv', index=False)

plt.figure()
sns.heatmap(X_VP.corr())
plt.savefig(path_root+'/Corr_plot_virtual_BGMM.png', dpi=600, bbox_inches='tight')
plt.close()

X_VP.columns = feature_names

plt.figure()
sns.heatmap(np.abs(X_corr-X_VP_corr), vmin=0, vmax=1, robust=True)
plt.savefig(path_root+'/Corr_plot_virtual_BGMM_diff.png', dpi=600, bbox_inches='tight')
plt.close()

plt.figure()
sns.heatmap(np.abs(X_cov-X_VP_cov), vmin=0, vmax=1, robust=True)
plt.savefig(path_root+'/Cov_plot_virtual_BGMM_diff.png', dpi=600, bbox_inches='tight')
plt.close()

y = 7
x = 5

plt.figure()
for i in range(0,np.size(X_VP,1)):
    kd_r = KernelDensity().fit(D.iloc[:,i].values.reshape(-1, 1))
    kd_v = KernelDensity().fit(X_VP.iloc[:,i].values.reshape(-1, 1))
    s = np.linspace(0,200)
    er = kd_r.score_samples(s.reshape(-1, 1))
    ev = kd_v.score_samples(s.reshape(-1, 1))
    KL_div = stats.entropy(er, ev)
    if(len(np.unique(X_VP.iloc[:,i])) == 2):
        vmr_r = np.std(D.iloc[:,i])/np.median(D.iloc[:,i])
        vmr_v = np.std(X_VP.iloc[:,i])/np.median(X_VP.iloc[:,i])
    elif(len(np.unique(X_VP.iloc[:,i])) == 1):
        vmr_r = (np.std(D.iloc[:,i])+0.000001)/(np.median(D.iloc[:,i])+0.000001)
        vmr_v = (np.std(X_VP.iloc[:,i])+0.000001)/(np.median(X_VP.iloc[:,i])+0.000001)
    else:
        vmr_r = np.std(D.iloc[:,i])/np.mean(D.iloc[:,i])
        vmr_v = np.std(X_VP.iloc[:,i])/np.mean(X_VP.iloc[:,i]) 
    vmr_diff = np.abs(vmr_r-vmr_v)
    
    plt.subplot(y,x,i+1)
    sns.kdeplot(D.iloc[:,i], color='k', label='real')
    sns.kdeplot(X_VP.iloc[:,i], color='m', label='virtual, cV='+str(np.around(vmr_diff,3)))
    plt.legend()
    plt.show()
plt.savefig(path_root+'/distributions_with_cV.png', dpi=600)
plt.close()
