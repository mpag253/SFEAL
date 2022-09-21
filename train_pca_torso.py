import os
import numpy as np
import pandas as pd
from scipy import sparse
import scipy.linalg as sla
#from time import sleep
from sys import exit

from src.sfeal import core as sf

#from subjects import subjects as list_of_subjects

from mlr_model_torso import EiMLR, EeMLR

# reload(sf)

sfmesh = sf.SSM_Mesh()
sfmodel = sf.SSM()

""" Some configurations. Modify as required. """
config = dict()
config["root dir"] = "/hpc/mpag253/Torso/surface_fitting"  # specify where the root directory for lung meshes are
config["fitted_lung_dir"] = "Lung/SurfaceFEMesh"
config["fitted_torso_dir"] = "Torso"
config["morphic original mesh path"] = "morphic_original"
config["morphic aligned mesh path"] = "morphic_aligned"

config["bodies"] = "LRT"  # can be L or R (left or right lung only) or LR (both lungs) or LRT (both lungs + torso)
config["morphic mesh name"] = "TorsoLung_fitted.mesh"
config["reference lung dir"] = os.path.join("Human_Aging","AGING001","EIsupine")
config["scale"] = True  # whether lung size is normalised

config["number of modes"] = "full"  # "full" needs work for meshes that don't exist ---
config["leave one out"] = "E" #None  # 'None' or a string ID for the leave one out (e.g. 'A')

# Read inputs from checklist
torso_list = np.array(pd.read_excel("/hpc/mpag253/Torso/torso_checklist.xlsx", usecols=range(13), engine='openpyxl'))
if config["leave one out"] is not None:
    loo_subject_index = torso_list[:, 12]==config["leave one out"]
    torso_list[loo_subject_index, 0] = 0
    config["loo subject"] = torso_list[loo_subject_index, 2][0]  # leave out out subject
    config["loo subject short"] = config["loo subject"].split('-')[-1]
    print('\nLEAVE ONE OUT:', config["leave one out"], '=', config["loo subject short"], '\n')
input_list = torso_list[torso_list[:, 0] == 1]
config["study"], config["subjects"], config["volume"] = input_list[:, 1:4].T
config["subjects for pca"] = []


def _get_mesh():
    """
    Gets the fitted meshes and convert them to morphic meshes.

    :return:
    meshes list containing morphic mesh paths.
    morphic reference_mesh used in _align() for Procrustes registration.

    """

    fitted_lung_dir = config["fitted_lung_dir"]
    subjects = config["subjects"]
    meshes = []
    not_exist_meshes = []

    for i, sub in enumerate(subjects):

        print("---subject: ", sub)

        study = config["study"][i]
        volume = config["volume"][i]
        
        full_path = os.path.join(config["root dir"], 'output', sub, volume, fitted_lung_dir)
        #/hpc/mpag253/Torso/surface_fitting/output/AGING001/EISupine/Torso/
        torso_path = os.path.join(config["root dir"], 'output', sub, volume, config["fitted_torso_dir"])
        output_path = sfmesh.convert_cm_mesh(full_path, torso_path, config["bodies"])   ##########

        if output_path is None:
            not_exist_meshes.append(sub)
            continue

        config["subjects for pca"].append(sub)
        if not os.path.exists(full_path + '/' + config["morphic original mesh path"] + '/' + config["morphic mesh name"]):
            sfmesh.generate_mesh(output_path, bodies=config["bodies"], save=True)

        for mesh_file in os.listdir(output_path):
            #if mesh_file.endswith(config["morphic mesh name"]):
            if mesh_file == config["morphic mesh name"]:
                meshes.append(os.path.join(output_path, mesh_file))

    #reference_mesh = '/hpc/mpag253/Torso/surface_fitting/Human_Aging/AGING025/EIsupine/Lung/SurfaceFEMesh/morphic_original/Lung_fitted.mesh'
    reference_mesh = os.path.join(config["root dir"], config["reference lung dir"], fitted_lung_dir, 'morphic_original', config["morphic mesh name"])

    return meshes, reference_mesh


def _align(m, r, scaling=False):
    """
    Method calling the align_mesh() method from SFEAL.core to Procrustes align all the meshes to the reference mesh
    specified above.

    :param m: meshes list from _get_mesh()
    :param r: reference_mesh from _get_mesh()
    :return: aligned_mesh_objects from SFEAL.core.align_mesh()
    """
    meshes = m
    reference_mesh = r

    aligned_mesh_objects = []

    for mesh in meshes:
        d, Z, tform, m, _ = sfmesh.align_mesh(reference_mesh, mesh, config["bodies"], scaling=scaling, reflection='best')
        aligned_mesh_objects.append(m)

    return aligned_mesh_objects


def _prepare_sfeal():
    """
    Preapring the meshes for PCA.

    :return: Prepared SFEAL model object.
    """
    #path = config["path"]
    #volume = config["volume"]
    fitted_lung_dir = config["fitted_lung_dir"]
    subjects = config["subjects for pca"]
    aligned_meshes = []

    for i, sub in enumerate(subjects):
    
        study = config["study"][i]
        volume = config["volume"][i]
        
        full_path = os.path.join(config["root dir"], 'output', sub, volume, fitted_lung_dir, config["morphic aligned mesh path"])
        
        if not os.path.exists(full_path):
            print('morphic_aligned directory does not exist for subject {}. Skipping...'.format(sub))
            continue
        for mesh_file in os.listdir(full_path):
            if mesh_file == config["morphic mesh name"]:
                aligned_meshes.append(os.path.join(full_path, mesh_file))

    #print(aligned_meshes)
    for mesh in range(len(aligned_meshes)):
        #print("mesh = ", mesh)
        sfmodel.add_mesh(aligned_meshes[mesh])

    return sfmodel, aligned_meshes


def _get_score(sfeal_model, mesh, aligned_mesh_names, pca_id):
    """
    Calculates mesh PCA scores and stores them in a csv file 'scores.csv' in the SFEAL module directory

    :param sfeal_model:
    :param aligned_mesh_names:
    :return:
    """
    pmesh = mesh
    sf = sfeal_model
    subject_names = sorted(aligned_mesh_names)
    m_distance = list()
    score_array = np.chararray((len(subject_names), sf.get_number_of_modes() + 1), itemsize=25)
    for i in range(len(subject_names)):
        score, ratio = sf.get_score(subject_names[i], pca_id)
        mah_distance = sf.get_mahalanobis()
        m_distance.append(mah_distance)
        score_array[i][0] = subject_names[i].split('/')[6]
        for j in range(sf.num_modes):
            score_array[i][j+1] = score['MODE {:>3d} SCORE'.format(j+1)]

    np.savetxt('output/scores/pca_scores_'+pca_id+'.csv', score_array.astype(np.str_), delimiter=',', fmt='%s')
    return m_distance


#def _read_file():
#    import pandas as pd
#    f = '/hpc/mpag253/Torso/SFEAL/input/temp_test_predict_subjects.csv'
#    df = pd.read_csv(f, header=0, delimiter=',')
#    return df
    
    
def generate_id():

    if config["scale"]:
        filename_scale_id = "S"
    else:
        filename_scale_id = "U"
        
    if config["leave one out"] is None:
        filename_LOO = ""
    else:
        filename_LOO = "_LOO-"+config["leave one out"]
        
    filename_ref_id = '-'.join(config["reference lung dir"].split('/')[1:3])
    
    pca_id = config["bodies"] + \
             '_' + filename_scale_id + \
             '_M' + str(config["number of modes"]) + \
             '_N' + str(len(config["subjects for pca"])) + \
             '_R-' + filename_ref_id + \
             filename_LOO
                         
    return pca_id


def main():

    # Retrieve meshes for PCA training
    meshes, ref_mesh = _get_mesh()
    aligned_mesh_objs = _align(meshes, ref_mesh, config["scale"])
    n_sbj_in_pca = len(config["subjects for pca"])

    # Create SFEAL and perform PCA
    sf, aligned_mesh_names = _prepare_sfeal()
    number_of_modes = n_sbj_in_pca-1 if config["number of modes"] == "full" else config["number of modes"]
    pca_id = generate_id()
    pmesh, _ = sf.pca_train(pca_id, num_modes=number_of_modes)
    sf.save_mesh_id()
       
    # Explained variance/variance ratios 
    #print("Explained variance:\n", sf.variance)
    #print("Explained variance ratio:")
    #[print(key,':',value) for key, value in sf.ratio.items()]
    #print("\n") 
    
    ################################################################################
    # OPTIONAL OUTPUTS
    ################################################################################
    
    
    # Saving the scores for the training samples: 
    export_scores = False
    if export_scores:
        _get_score(sf, pmesh, aligned_mesh_names, pca_id)
        
    
    # Export mode shapes 
    export_mode_shapes = False
    if export_mode_shapes:  
        mean_shape = sf.get_sampled_nodes(pmesh, np.zeros([5])).flatten()
        n_dof = len(mean_shape)
        weights = np.eye(number_of_modes)
        modeshapes = np.empty([n_dof, number_of_modes])
        for i in range(number_of_modes):
            modeshape = sf.get_sampled_nodes(pmesh, weights[i]).flatten()
            modeshapes[:, i] = modeshape - mean_shape
        np.save('output/mode_shapes/pca_modeshapes_'+pca_id+'.npy', modeshapes)
        #with np.printoptions(precision=3, suppress=True):
        #    print("\n\tMean:\n", mean_shape)
        #with np.printoptions(precision=3, suppress=True):
        #    print("\n\tmodeshapes:\n", modeshapes)   
        
    
    # Saving the mean shape to cmiss files (i.e. ipnode)
    export_mean_shape = True
    if export_mean_shape:
        weights = np.zeros(number_of_modes)
        if config["bodies"] == "LRT":
            sf.export_to_cm(pmesh, weights, name='pca_'+pca_id+'/sample_mean', body='T', show_mesh=False)
            sf.export_to_cm(pmesh, weights, name='pca_'+pca_id+'/sample_mean', body='L', show_mesh=False)
            sf.export_to_cm(pmesh, weights, name='pca_'+pca_id+'/sample_mean', body='R', show_mesh=False)
        elif config["bodies"] == "LR":
            sf.export_to_cm(pmesh, weights, name='pca_'+pca_id+'/sample_mean', body='L', show_mesh=False)
            sf.export_to_cm(pmesh, weights, name='pca_'+pca_id+'/sample_mean', body='R', show_mesh=False)


    # For projecting non-training samples onto the trained PCA:
    # not updated by MP ???
    generate_sample_projections = False
    if generate_sample_projections:
        pr_path = "/hpc/mpag253/Torso/surface_fitting/Human_Aging"
        pr_path_1 = "/EIsupine/Lung/SurfaceFEMesh/morphic_aligned/Lung_fitted.mesh"
        
        projected_weights = dict()
        for project_subject in os.listdir(pr_path):
           pr_sub_path = pr_path + "/" + project_subject + pr_path_1
           if os.path.exists(pr_sub_path):
               print("Subject = {}".format(project_subject))
               w, r = sf.project_new_mesh(pr_sub_path)
               projected_weights[project_subject] = w
        
        weight_list = list()
        for subjects in projected_weights.keys():
           weight_list.append(subjects)
           for modes in sorted(projected_weights[subjects]):
               weight_list.append(projected_weights[subjects][modes])
        
        a = np.asarray(weight_list, dtype=object)
        b = a.reshape(n_sbj_in_pca, number_of_modes+1)
        np.savetxt('/people/mpag253/Desktop/subject_weights.txt', b, fmt='%s')


    # For exporting +/-2.5 standard deviation mode weights as shapes:
    export_mode_shapes = False
    if export_mode_shapes:
        #modes = np.arange(number_of_modes) + 1
        modes = np.arange(10) + 1
        weights = np.zeros(number_of_modes)
        for m in modes:
            for w in np.linspace(-2.5, 2.5, 2):
                weights[m-1] = w
                print(weights)
                if w == -2.5:
                    config["export_name"] = 'pca_'+pca_id + '/mode{:02d}_neg2p5'.format(m)
                else:
                    config["export_name"] = 'pca_'+pca_id + '/mode{:02d}_pos2p5'.format(m)
                sf.export_to_cm(pmesh, weights, name=config["export_name"], body='L', show_mesh=False)
                sf.export_to_cm(pmesh, weights, name=config["export_name"], body='R', show_mesh=False)
                sf.export_to_cm(pmesh, weights, name=config["export_name"], body='T', show_mesh=False)
                weights = np.zeros(number_of_modes)

 
    # For LEAVE-ONE-OUT MLR shape predictions
    generate_loo_prediction = True
    if generate_loo_prediction and (config["leave one out"] is not None):
        print("\n\t"+"="*41+"\n\n\tGENERATING MLR PREDICTIONS...")
        
        # Retrieve the LOO subject data
        f = '/hpc/mpag253/Torso/SFEAL/input/pca_subject_data.csv'
        pca_subject_data = pd.read_csv(f, skiprows=0, header=1, delimiter=',')
        loo_subject_data = pca_subject_data[pca_subject_data['Subject'] == config["loo subject short"]]

        # Retrieve the MLR model for the LOO subject
        import pickle
        with open('output/mlr_models/pca_mlr_model_'+pca_id+'.pkl', 'rb') as f:
            mlr_model = pickle.load(f)
        
        # EI
        ei = EiMLR(mlr_model, loo_subject_data) 
        ei_weights = ei.predict_modes()
        export_name = 'pca_'+pca_id+'/predict_'+config["loo subject short"]
        sf.export_to_cm(pmesh, ei_weights, name=export_name, body='L', show_mesh=False)
        sf.export_to_cm(pmesh, ei_weights, name=export_name, body='R', show_mesh=False)
        sf.export_to_cm(pmesh, ei_weights, name=export_name, body='T', show_mesh=False)
        #ei_mean = sf.get_sampled_nodes(pmesh, ei_weights).flatten()
        #with np.printoptions(precision=2, suppress=True):
        #    print("\n\tPredicted mean nodes:\n\t", ei_mean)
        
        ## EE
        #ee = EeMLR(pfts['age'], pfts['frc'], pfts['rv'], pfts['rvtlc'], pfts['dlco'], pfts['fev1'], pfts['pefr'], pfts['vc'])
        #eem1 = ee.predict_m1()
        #eem2 = ee.predict_m2()
        #eem3 = ee.predict_m3()
        #weights_ee = [eem1, eem2, eem3]
        #export_name = 'pca_predict_'+pfts['subject']+'-EE_from_'+pca_id
        #sf.export_to_cm(pmesh, weights_ee, name=export_name, body='L', show_mesh=False)
        #sf.export_to_cm(pmesh, weights_ee, name=export_name, body='R', show_mesh=False)
        #sf.export_to_cm(pmesh, weights_ee, name=export_name, body='T', show_mesh=False)
        
        print("\n\tSAVED PREDICTED MEAN MESH TO:\n\t"+export_name+"\n\n\t"+"="*41+"\n")
            
            
    # scores, mahalanobis_distance = _get_score(sf, aligned_mesh_names)

    return sf, pmesh


if __name__ == "__main__":
    sf, pmesh = main()
