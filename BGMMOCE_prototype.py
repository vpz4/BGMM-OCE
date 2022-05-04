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
import re
from sklearn.impute import KNNImputer
from scipy import stats
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KernelDensity
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def get_clustering_score(X, k, sc_type):
    print('starting k={}'.format(k));
    
    model = SpectralClustering(k,
                                eigen_solver='lobpcg',
                                # gamma=chi2.pdf(2,1,1/k),
                                # gamma=np.exp(-k),
                                assign_labels='discretize',
                                random_state=0
                                );
    model.fit(X);
    labels = model.labels_;
    
    if(sc_type == 1):
        score = davies_bouldin_score(X, labels);
    elif(sc_type == 2):
        score = silhouette_score(X, labels);
    else:
        score = calinski_harabasz_score(X, labels);
    
    return score;

def opt(X):
    for g in np.array(range(0,50)):
        scores  = [];
        for k in np.array(range(2,20)):
            model = SpectralClustering(k,
                                       # eigen_solver='lobpcg',
                                       # gamma=chi2.pdf(k,k,1/k),
                                       gamma=g/1000,
                                       assign_labels='discretize');
            model.fit(X);
            labels = model.labels_;
            score = silhouette_score(X, labels);
            scores.append(score);
            
        plt.plot(np.array(range(2,20)), scores, '-o');
        plt.legend('g='+str(g));
        plt.show();
    return score;

def find_optimal_N(s):
    N = np.where(s == np.min(s))[0][0];
    return N;

def find_optimal_N2(s):
    N = np.where(s == np.max(s))[0][0];
    return N;

def find_optimal_N3(s, th):
    N = 0;
    flag = 0;
    for i in range(len(s)-1):
        s1 = len(re.search('\d+\.(0*)', str('{0:.10f}'.format(s[i]))).group(1));
        s2 = len(re.search('\d+\.(0*)', str('{0:.10f}'.format(s[i+1]))).group(1));
        if(np.abs(s1-s2) >= th):
            N = i;
            flag = 1;
            break;
    
    if(flag != 1):
        N = 6;
    
    return N+1;

#BGMM
os.chdir(os.getcwd());

T = pd.read_csv('HCM/2020-11-09-live-clinical_curated_data.csv');
features = ['age', 'sex', 'nyhaClass', 'systolicPressure', 'diastolicPressure', 
            'syncope', 'heartMurmur', 'eflv', 'lvids', 'lvidd', 'ivsd', 'plwd',
            'svlv', 'lvotMaxPg', 'ee', 'bmi', #good
            'la', 'alt', 'ao', 'aorticValve']; #good
D = T[features];
feature_names = list(D.columns);

imputer = KNNImputer(n_neighbors=1, weights='uniform', metric='nan_euclidean');
D = imputer.fit_transform(D);
D = pd.DataFrame(columns=feature_names, data=D);

features_simple = features;
D.columns = features_simple;

X_corr = D.corr(method='pearson');
X_cov = D.cov();
cV_real = D.std()/D.mean();

plt.figure(figsize=(18, 13));
sns.heatmap(D.corr());
plt.savefig('new/Corr_plot_real_V2.png', dpi=600, bbox_inches='tight');
plt.close();
    
scaler = RobustScaler();
data = scaler.fit_transform(D);

data = pd.DataFrame(data=data, columns=D.columns);

n_max = 21; #max number of Gaussian components
N_max = 31000; #max number of virtual patients
N_min = 1000;
step_pat = 1000; #step of virtual patients
n_comps = np.array(range(2,n_max)); #number of Gaussian components/clusters to evaluate
num_patients = np.array(range(N_min,N_max,step_pat)); #number of patients to evaluate

#apply spctral clustering and evaluate the clusters
start = time.time();
kms_scores = [(k, get_clustering_score(data, k, 1)) for k in n_comps];

scores = [kms_scores[i][1] for i in range(0,len(kms_scores))];
ncomps = [kms_scores[i][0] for i in range(0,len(kms_scores))];

scores = np.array(scores, dtype=float);
ncomps = np.array(ncomps, dtype=int);

#extract optimal number of clusters
opt_num = find_optimal_N2(scores)+2;
end = time.time();
print(end-start);
  
weight_concentration_prior = np.exp(-opt_num);

#plot clustering oriented figures
plt.figure(1, figsize=(18,10));
sns.set(font_scale=2);
plt.plot(ncomps,scores,'-o');
plt.xlabel('components');
plt.ylabel('avg. DB score');
plt.xticks(np.array(range(2,np.max(ncomps)+2,2)), np.array(range(2,np.max(ncomps)+2,2)));
plt.legend();
plt.show();
plt.savefig('new/avg_DB_scores_V2.png', dpi=600, bbox_inches='tight');
plt.close();

#print results
print("Optimal number of components according to highest avg. DB index is:", int(opt_num));
print("");

# fit BGMM
start = time.time();
g = BayesianGaussianMixture(n_components=opt_num,
                            covariance_type='full',
                            random_state=0,
                            weight_concentration_prior=weight_concentration_prior);
g.fit(data);
end = time.time();
print(end-start);

gofss = [];
KL_divss = [];
scoress = [];
cof_dif_scoress = [];
cov_dif_scoress = [];
std_cof_dif_scoress = [];
std_cov_dif_scoress = [];
weights_opts = [];
opts = [];
times = [];
CORR = [];
COV = [];
cV_diff = [];
cV_virtual = [];
vmrrr = [];
        
for num_patient in num_patients:
    path_root = 'new/'+str(num_patient);
    
    if(not os.path.isdir(path_root)):
        os.makedirs(path_root);
    print("Number of virtual patients =", num_patient);
    
    #backprojection
    start = time.time();
    Xnew = g.sample(num_patient)[0];
    Xnew2 = np.around(Xnew);
    Xnew3 = scaler.inverse_transform(Xnew2);
    X_VP = pd.DataFrame(columns = D.columns, data = Xnew3);
    end = time.time();
    times.append(end-start);
    print("Elapsed time = ", str(end-start));
    print("");
    
    aa = [];
    p_aa = [];
    KL_divv = [];
    vmrr = [];
    i = 0;
    for s in list(D.columns):
        [a, p_a] = stats.ks_2samp(D.iloc[:,i], X_VP.iloc[:,i]);
        aa.append(a);
        p_aa.append(p_a);
        kd_r = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(D.iloc[:,i].values.reshape(-1, 1));
        kd_v = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X_VP.iloc[:,i].values.reshape(-1, 1));
        s = np.linspace(0,200);
        er = kd_r.score_samples(s.reshape(-1, 1));
        ev = kd_v.score_samples(s.reshape(-1, 1));
        KL_div = stats.entropy(er, ev);
        vmr_r = np.std(D.iloc[:,i])/np.mean(D.iloc[:,i]);
        vmr_v = np.std(X_VP.iloc[:,i])/np.mean(X_VP.iloc[:,i]);
        vmr_diff = np.abs(vmr_r-vmr_v);
        KL_divv.append(KL_div);
        vmrr.append(vmr_diff);
        i += 1;
        
    X_VP = pd.DataFrame(columns = D.columns, data = Xnew3);
    X_VP.to_excel(path_root+'/Vpop_'+str(num_patient)+'_V2.xlsx', index=False);
    
    X_VP_corr = X_VP.corr(method='pearson');
    X_VP_corr.fillna(1, inplace=True);
    X_diff = np.abs(X_corr-X_VP_corr);
    diff_metric = X_diff.values[np.triu_indices_from(X_diff.values,1)].mean();
    std_metric  = X_diff.values[np.triu_indices_from(X_diff.values,1)].std();
    CORR.append(X_diff);
 
    X_VP_cov = X_VP.cov();
    X_diff2 = np.abs(X_cov-X_VP_cov);
    diff_metric2 = X_diff2.values[np.triu_indices_from(X_diff2.values,1)].mean();
    std_metric2  = X_diff2.values[np.triu_indices_from(X_diff2.values,1)].std();
    COV.append(X_diff2);
    
    cV_virtual.append(X_VP.std()/X_VP.mean());
    cV_diff.append(np.abs((D.std()/D.mean())-(X_VP.std()/X_VP.mean())));
    
    gofss.append(np.mean(aa));
    KL_divss.append(np.mean(KL_divv));
    scoress.append(g.score(data));
    cof_dif_scoress.append(diff_metric);
    cov_dif_scoress.append(diff_metric2);
    std_cof_dif_scoress.append(std_metric);
    std_cov_dif_scoress.append(std_metric2);
    weights_opts.append(weight_concentration_prior);
    
    vmrrr.append(vmrr);
    
    plt.figure(figsize=(18, 13));
    sns.heatmap(X_VP.corr());
    plt.savefig(path_root+'/Corr_plot_virtual_BGMM_V2.png', dpi=600, bbox_inches='tight');
    plt.close();
    
    X_VP.columns = features_simple;
    
    plt.figure(figsize=(18, 13));
    sns.heatmap(np.abs(X_corr-X_VP_corr), vmin=0, vmax=1, robust=True);
    plt.savefig(path_root+'/Corr_plot_virtual_BGMM_diff_V2.png', dpi=600, bbox_inches='tight');
    plt.close();

    plt.figure(figsize=(18, 13));
    sns.heatmap(np.abs(X_cov-X_VP_cov), vmin=0, vmax=1, robust=True);
    plt.savefig(path_root+'/Cov_plot_virtual_BGMM_diff_V2.png', dpi=600, bbox_inches='tight');
    plt.close();

    plt.rcParams['font.size'] = '11';
    plt.figure(figsize=(22, 22));
    y = 4;
    x = int(len(features)/y);
    for i in range(0,np.size(X_VP,1)):
        kd_r = KernelDensity().fit(D.iloc[:,i].values.reshape(-1, 1));
        kd_v = KernelDensity().fit(X_VP.iloc[:,i].values.reshape(-1, 1));
        s = np.linspace(0,200);
        er = kd_r.score_samples(s.reshape(-1, 1));
        ev = kd_v.score_samples(s.reshape(-1, 1));
        KL_div = stats.entropy(er, ev);
        vmr_r = np.std(D.iloc[:,i])/np.mean(D.iloc[:,i]);
        vmr_v = np.std(X_VP.iloc[:,i])/np.mean(X_VP.iloc[:,i]);
        vmr_diff = np.abs(vmr_r-vmr_v);
        
        plt.subplot(y,x,i+1);
        sns.kdeplot(D.iloc[:,i], color='k', label='real');
        sns.kdeplot(X_VP.iloc[:,i], color='m', label='virtual, cV='+str(np.around(vmr_diff,3)));
        plt.legend();
        plt.show();
        plt.tight_layout();
    plt.savefig(path_root+'/distributions_with_cV_V2.png', dpi=600, bbox_inches='tight');
    plt.close();

times_DF = pd.DataFrame(data={'VP_num':num_patients, 'time':times});
times_DF.to_excel('new/times_BGMM_V2.xlsx', index=False);

#calculate intra and inter measures
inter_correlation_dif = np.mean(np.mean(CORR,2));
intra_correlation_dif = np.mean(np.mean(CORR,2),1);
inter_cov_dif = np.mean(np.mean(COV,2));
intra_cov_dif = np.mean(np.mean(COV,2),1);

plt.figure(figsize=[20,12])
plt.subplot(221)
plt.plot(scores, '-o')
plt.ylabel('DBS')
plt.xticks(np.array(range(0,len(num_patients)+1,2)),
            np.array(range(1,len(num_patients)+2,2))*1000);
plt.xlabel('number of virtual patients');
plt.subplot(222)
plt.plot(KL_divss, '-o')
plt.ylabel('KL-divergence difference')
plt.xticks(np.array(range(0,len(num_patients)+1,2)),
            np.array(range(1,len(num_patients)+2,2))*1000);
plt.xlabel('number of virtual patients');
plt.subplot(223)
plt.plot(np.mean(np.mean(CORR,2),0), '-o')
plt.ylabel('Inter-correlation difference')
plt.xticks(np.array(range(0,len(num_patients)+1,2)),
            np.array(range(1,len(num_patients)+2,2))*1000);
plt.xlabel('number of virtual patients');
plt.subplot(224)
plt.plot(np.mean(np.mean(CORR,2),1), '-o')
plt.ylabel('Intra-correlation difference')
plt.xticks(np.array(range(0,len(num_patients)+1,2)),
            np.array(range(1,len(num_patients)+2,2))*1000);
plt.xlabel('number of virtual patients');
plt.savefig('new/total_specs_V2.png', dpi=300, bbox_inches='tight');

plt.figure(figsize=[20,8])
sns.set(font_scale=1.5);
plt.subplot(121)
plt.plot(ncomps,scores,'-o');
plt.xlabel('clusters');
plt.ylabel('DBS');
plt.xticks(rotation = 45);
plt.xticks(np.array(range(2,np.max(ncomps)+2,2)), np.array(range(2,np.max(ncomps)+2,2)));
plt.subplot(122)
plt.plot(np.mean(np.mean(CORR,2),1), '-o')
plt.ylabel('Intra-correlation difference')
plt.xticks(np.array(range(0,len(num_patients)+1,3)),
            np.array(range(1,len(num_patients)+2,3))*1000);
plt.xlabel('number of virtual patients');
plt.xticks(rotation = 45);
# plt.plot(scoress, '-o')
# plt.ylabel('BGMM score')
# plt.xticks(np.array(range(0,len(num_patients)+1,3)),
#             np.array(range(1,len(num_patients)+2,3))*1000);
# plt.xlabel('number of virtual patients');
plt.savefig('new/total_specs_v3_V2.png', dpi=300, bbox_inches='tight');

######################################################################################################
# plt.rcParams['font.size'] = '10';
# plt.figure(figsize=(22, 22));
# plt.subplot(2,2,1);
# plt.plot(gofsss, '-o');
# plt.ylabel('goodness of fit');
# plt.xlabel('number of patients');
# plt.xticks(np.array(range(0,len(num_patients)+1,2)),np.array(range(1,len(num_patients)+2,2))*1000);
# plt.show();
# plt.subplot(2,2,2);
# plt.plot(cof_dif_scoresss, '-o');
# plt.ylabel('intra-correlation');
# plt.xlabel('number of patients');
# plt.xticks(np.array(range(0,len(num_patients)+1,2)),np.array(range(1,len(num_patients)+2,2))*1000);
# plt.show();
# plt.subplot(2,2,3);
# plt.plot(KL_divsss, '-o');
# plt.ylabel('KL divergence');
# plt.xticks(np.array(range(0,len(num_patients)+1,2)),np.array(range(1,len(num_patients)+2,2))*1000);
# plt.show();
# plt.subplot(2,2,4);
# plt.plot(scoresss, '-o');
# plt.ylabel('BGMM score');
# plt.xticks(np.array(range(0,len(num_patients)+1,2)),np.array(range(1,len(num_patients)+2,2))*1000);
# plt.show();
# plt.savefig('new/Statistics_across_the_virtual_patients.png', dpi=600, bbox_inches='tight');