import numpy as np
from scipy import sparse

pca_id = 'LRT_S_Mfull_N81_R-AGING001-EIsupine'
n_samples = 10000

fname = 'pca_condcov_est-'+str(n_samples)+'_'+pca_id+'.npz'
spmat = sparse.load_npz('output/covariance/'+fname)
array = spmat.toarray() 

print('\nConditional covariance', np.shape(array))
print('({})\n\n'.format(fname), array, '\n')



