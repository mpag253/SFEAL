import joblib
import numpy as np
from scipy import sparse
import pickle

root = '/hpc/mpag253/Torso/SFEAL/'

#pca_id = 'LR_U_M5_N83_R-AGING025-EIsupine'
#pca_id = 'LR_U_Mfull_N83_R-AGING025-EIsupine'
#pca_id = 'LR_S_M5_N83_R-AGING025-EIsupine'
#pca_id = 'LRT_U_M5_N76_R-AGING025-EIsupine'
pca_id = 'LRT_S_M5_N76_R-AGING025-EIsupine'
#pca_id = 'LRT_S_M68_N76_R-AGING025-EIsupine'

save_covariance = False
save_mean_and_cov = True

# Load PCA
pca = joblib.load('output/pca_model_'+pca_id+'.sfeal')

# PCA object:
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

print("\n\nPCA Components ",               np.shape(pca.components_),               ":\n\n", pca.components_)
print("\n\nPCA Explained Variance ",       np.shape(pca.explained_variance_),       ":\n\n", pca.explained_variance_)
print("\n\nPCA Explained Variance Ratio ", np.shape(pca.explained_variance_ratio_), ":\n\n", pca.explained_variance_ratio_)
cumvar = np.cumsum(pca.explained_variance_ratio_)
print("\n\nPCA Cumulative Explained Variance ", np.shape(cumvar),                   ":\n")
print(', '.join('{:.4f}'.format(k) for k in cumvar))
print("\n\nPCA Total Explained Variance:", np.sum(pca.explained_variance_ratio_))
print("\n\nPCA Singular Values ",          np.shape(pca.singular_values_),          ":\n\n", pca.singular_values_)
print("\n\nPCA Mean ",                     np.shape(pca.mean_),                     ":\n\n", pca.mean_)
print("\n\nPCA Noise Variance:", pca.noise_variance_)

pca_cov = pca.get_covariance()
print("\n\nPCA Covariance ",               np.shape(pca_cov),                       ":\n\n", pca_cov)

if save_covariance:
    spm = sparse.csr_matrix(pca_cov)
    sparse.save_npz(root+'output/covariance/pca_cov_'+pca_id+'.npz', spm)
    print("\n\nPCA covariance matrix saved.") 

# attempting to reproduce covariance manually...
#unexp_cov = pca.noise_variance_*np.eye(pca.n_features_)
#print("\n\nPCA Covariance explained",      np.shape(pca_cov),                       ":\n\n", pca_cov - unexp_cov)
#n_subj = pca.n_samples_
##eigvals = np.diag(np.square(pca.explained_variance_)) / (n_subj-1)
#eigvals = np.diag(pca.explained_variance_) #/ (n_subj-1)
#pca_cov_test = np.matmul(np.matmul(pca.components_.T, eigvals), pca.components_) #+ unexp_cov
#print("\n\nPCA Covariance TEST explained",  np.shape(pca_cov_test),                  ":\n\n", pca_cov_test)

loadfile = open("latest_pca_data_array.pkl", "rb")
raw_data = pickle.load(loadfile)
loadfile.close()
mean_data = np.mean(raw_data, axis=0)
print("\n\nData Mean",  np.shape(mean_data),                  ":\n\n", mean_data)
zeroed_data = raw_data - mean_data
N = np.shape(zeroed_data)[0]
print("\n\nN:", N)
cov_data = np.matmul(zeroed_data.T, zeroed_data) / (N-1)
print("\n\nData Covariance",  np.shape(cov_data),                  ":\n\n", cov_data)

wrong_cov_data = np.matmul(raw_data.T, raw_data) / (N-1)
print("\n\nWRONG Data Covariance",  np.shape(cov_data),                  ":\n\n", cov_data)

if save_mean_and_cov:
    dumpfile1 = root+'output/latest_data_mean.npy'
    dumpfile2 = root+'output/latest_data_cov.npz'
    #np.save(dumpfile1, mean_data)
    np.save(dumpfile1, pca.mean_)
    spm = sparse.csr_matrix(cov_data)
    sparse.save_npz(dumpfile2, spm)

print("\n\n")









