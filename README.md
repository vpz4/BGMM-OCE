# BGMM-OCE

This is an open-source repository which contains the implementation of the BGMM-OCE (Bayesian inference-based Gaussian Mixture Models with Optimal Components Estimation) algorithm.

**Related publication**
- Pezoulas, Vasileios C., et al. "Bayesian inference-based Gaussian mixture models with optimal components estimation towards large-scale synthetic data generation for in silico clinical trials." IEEE Open Journal of Engineering in Medicine and Biology 3 (2022): 108-114. (https://ieeexplore.ieee.org/document/9794449).

**2024 Updates**
- The optimal component estimation (OCE) process has been updated using the MiniSOM clustering algorithm instead of the spectral clustering algorithm to avoid errors durign the esstimation of Laplacian matrices.
- Promt request from the user regarding the path location of the real data and the number of virtual patients to be generated.
