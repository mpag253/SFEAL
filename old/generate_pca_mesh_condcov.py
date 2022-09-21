import joblib
import numpy as np





pca_id = 'LRT_S_Mfull_N81_R-AGING001-EIsupine'
n_mlr_modes = 5




# Load the pca 
pca = joblib.load('output/models/pca_model_'+pca_id+'.sfeal')
# Retrieve the mean
pca_mean = pca.mean_
print('\npca_mean:', np.shape(pca_mean))
print(pca_mean)
# Retrieve the components
pca_components = pca.components_.T
print('\npca_components:', np.shape(pca_components))
print(pca_components)
# Retrieve the singular values
pca_singvals = pca.singular_values_
print('\npca_singvals:', np.shape(pca_singvals))
print(pca_singvals)

## (test the components)
#component_2norms = np.linalg.norm(pca_components, ord=2, axis=0)
#print('\ncomponent_2norms:', np.shape(component_2norms))
#print(component_2norms)


## Get the components relevant for the mlr model
#mlr_components = pca_components[:, :n_mlr_modes]
#print('\nmlr_components:', np.shape(mlr_components))
#print(mlr_components)
## Get the singular values relevant for the mlr model
#mlr_singvals = pca_singvals[:n_mlr_modes]
#print('\nmlr_singvals:', np.shape(mlr_singvals))
#print(mlr_singvals)
## Caulculate the mode shapes
#mlr_modeshapes = manual_modeshapes #np.multiply(mlr_components, mlr_singvals*0.1118)
#print('\nmlr_modeshapes:', np.shape(mlr_modeshapes))
#print(mlr_modeshapes)

# load in manually generated mode shapes
full_modeshapes = np.loadtxt('manual_modeshapes.txt')


# Import conditoinal covariance of weights 
# (generated from EIT Project/SSM/torso_pca_mlr.py)
mlr_condcov_file = "input/conditional_covariance/mlr_M{:d}_condcov_{}.csv".format(n_mlr_modes, pca_id)
mlr_condcov = np.loadtxt(mlr_condcov_file, delimiter=",")
print('\nmlr_condcov:', np.shape(mlr_condcov))
print(mlr_condcov)
# full
full_condcov = np.eye(len(pca_singvals))
full_condcov[:n_mlr_modes, :n_mlr_modes] = mlr_condcov

# Calculte the mesh conditional covariance
#mesh_condcov = np.matmul(np.matmul(mlr_modeshapes, mlr_condcov), mlr_modeshapes.T)
mesh_condcov = np.matmul(np.matmul(full_modeshapes, full_condcov), full_modeshapes.T)
print('mesh_condcov:', np.shape(mesh_condcov))
print(mesh_condcov)

# Retrieve the PCA covariance
pca_cov = pca.get_covariance()
print('pca_cov:', np.shape(pca_cov))
print(pca_cov)

####################################
## reproduce a mesh shape using built-in and then manually to compare  
#weights_test = [1.0, -0.8, 0.6, -0.4, 0.2]
#nodes_test = pca_mean + np.matmul(mlr_modeshapes, weights_test)
#nodes_test = nodes_test.reshape([int(len(nodes_test)/12), 12])
#with np.printoptions(precision=2, suppress=True):
#    print("\n\tMean:\n\t", pca_mean)
#    print("\n\tNodes test:\n\t", nodes_test)
#np.savetxt('example_nodes_test.txt', nodes_test, fmt='%.4e')







