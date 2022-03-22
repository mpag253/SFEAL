import os
import numpy as np
import pandas as pd
#from time import sleep

from src.sfeal import core as sf

#from subjects import subjects as list_of_subjects

# from mlr import EiMLR, EeMLR

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
config["subjects for pca"] = []

#config["bodies"] = "LR"  # can be L or R (left or right lung only) or LR (both lungs) or LRT (both lungs + torso)
#config["morphic mesh name"] = "Lung_fitted.mesh"
config["bodies"] = "LRT"  # can be L or R (left or right lung only) or LR (both lungs) or LRT (both lungs + torso)
config["morphic mesh name"] = "TorsoLung_fitted.mesh"

config["reference lung dir"] = os.path.join("Human_Aging","AGING025","EIsupine")

config["number of modes"] = 5 # "full"  # 68

config["scale"] = True  # whether lung size is normalised

""" Read inputs from checklist """
input_list = np.array(pd.read_excel("/hpc/mpag253/Torso/torso_checklist.xlsx", skiprows=0, usecols=range(5), engine='openpyxl'))
input_list = input_list[input_list[:, 0] == 1]
config["study"], config["subjects"], config["volume"] = input_list[:, 1:4].T


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
        
        full_path = os.path.join(config["root dir"], study, sub, volume, fitted_lung_dir)
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
        
        full_path = os.path.join(config["root dir"], study, sub, volume, fitted_lung_dir, config["morphic aligned mesh path"])
        
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

    np.savetxt('output/pca_scores_'+pca_id+'.csv', score_array.astype(np.str_), delimiter=',', fmt='%s')
    return m_distance


def _read_file():
    import pandas as pd
    f = '/people/mosa004/Desktop/predict_subjects_for_Hari.csv'
    df = pd.read_csv(f, header=0, delimiter=',')
    return df
    
    
def generate_id():
    if config["scale"]:
        filename_scale_id = "S"
    else:
        filename_scale_id = "U"
    filename_ref_id = '-'.join(config["reference lung dir"].split('/')[1:3])
    pca_id = config["bodies"] + \
                           '_' + filename_scale_id + \
                          '_M' + str(config["number of modes"]) + \
                          '_N' + str(len(config["subjects for pca"])) + \
                         '_R-' + filename_ref_id
    return pca_id


def main():
    meshes, ref_mesh = _get_mesh()
    aligned_mesh_objs = _align(meshes, ref_mesh, config["scale"])
    number_of_subjects_in_pca = len(config["subjects for pca"])

    """ Create SFEAL and perform PCA """
    sf, aligned_mesh_names = _prepare_sfeal()
    number_of_modes = number_of_subjects_in_pca - 1 if config["number of modes"] == "full" else config["number of modes"]
    pca_id = generate_id()
    pmesh, _ = sf.pca_train(pca_id, num_modes=number_of_modes)
    sf.save_mesh_id()

    # Saving the scores for the training samples: 
    _get_score(sf, pmesh, aligned_mesh_names, pca_id)
    
    # Saving the mean shape to cmiss files (i.e. ipnode)
    weights = np.zeros(number_of_modes)
    if config["bodies"] == "LRT":
        sf.export_to_cm(pmesh, weights, name='pca_mean_'+pca_id, body='T', show_mesh=False)
        sf.export_to_cm(pmesh, weights, name='pca_mean_'+pca_id, body='L', show_mesh=False)
        sf.export_to_cm(pmesh, weights, name='pca_mean_'+pca_id, body='R', show_mesh=False)
    elif config["bodies"] == "LR":
        sf.export_to_cm(pmesh, weights, name='pca_mean_'+pca_id, body='L', show_mesh=False)
        sf.export_to_cm(pmesh, weights, name='pca_mean_'+pca_id, body='R', show_mesh=False)


    # # For projecting non-training samples onto the trained PCA:
    #
    # pr_path = "/hpc/mpag253/Torso/surface_fitting/Human_Aging"
    # pr_path_1 = "/EIsupine/Lung/SurfaceFEMesh/morphic_aligned/Lung_fitted.mesh"
    #
    # projected_weights = dict()
    # for project_subject in os.listdir(pr_path):
    #    pr_sub_path = pr_path + "/" + project_subject + pr_path_1
    #    if os.path.exists(pr_sub_path):
    #        print("Subject = {}".format(project_subject))
    #        w, r = sf.project_new_mesh(pr_sub_path)
    #        projected_weights[project_subject] = w
    #
    # weight_list = list()
    # for subjects in projected_weights.keys():
    #    weight_list.append(subjects)
    #    for modes in sorted(projected_weights[subjects]):
    #        weight_list.append(projected_weights[subjects][modes])
    #
    # a = np.asarray(weight_list, dtype=object)
    # b = a.reshape(number_of_subjects_in_pca, number_of_modes+1)
    # np.savetxt('/people/mpag253/Desktop/subject_weights.txt', b, fmt='%s')

    # # For exporting mode weights as shapes:
    #
    # modes = [1, 2, 3, 4]
    # weights = [0, 0, 0, 0]
    #
    # for m in modes:
    #     for w in np.linspace(-2.5, 2.5, 2):
    #         if m == 1:
    #             weights[0] = w
    #         elif m == 2:
    #             weights[1] = w
    #         elif m == 3:
    #             weights[2] = w
    #         elif m == 4:
    #             weights[3] = w
    #
    #         print weights
    #
    #         if w == -2.5:
    #             config["export_name"] = 'mode_{}_N25_no_scale'.format(m)
    #         else:
    #             config["export_name"] = 'mode_{}_P25_no_scale'.format(m)
    #
    #         # MP: note need to change from "lung="
    #         sf.export_to_cm(pmesh, weights, name=config["export_name"], lung='L', show_mesh=False)
    #         sf.export_to_cm(pmesh, weights, name=config["export_name"], lung='R', show_mesh=False)
    #
    #         weights = [0, 0, 0, 0]

    # config["export_name"] = 'mode_1_N25_no_scale'
    # sf.export_to_cm(pmesh, weights, name=config["export_name"], lung='L', show_mesh=False)

    # # For (multiple linear regression) ...
    #
    # pft_df = _read_file()
    # pfts = dict()
    # for subject, row in pft_df.iterrows():
    #     pfts['subject'] = row['subjects']
    #     pfts['age'] = row['age']
    #     pfts['bmi'] = row['bmi']
    #     pfts['fvc'] = row['fvc']
    #     pfts['dlco'] = row['dlco']
    #     pfts['tlc'] = row['tlc']
    #     pfts['rv'] = row['rv']
    #     pfts['frc'] = row['frc']
    #     pfts['fev1'] = row['fev1']
    #     pfts['vc'] = row['vc']
    #     pfts['pefr'] = row['pefr']
    #     pfts['rvtlc'] = row['rvtlc']
    #
    #     ei = EiMLR(pfts['age'], pfts['fvc'], pfts['dlco'], pfts['bmi'], pfts['rvtlc'], pfts['tlc'])
    #     ee = EeMLR(pfts['age'], pfts['frc'], pfts['rv'], pfts['rvtlc'], pfts['dlco'], pfts['fev1'], pfts['pefr'],
    #                pfts['vc'])
    #
    #     eim1 = ei.predict_m1()
    #     eim2 = ei.predict_m2()
    #     eim3 = ei.predict_m3()
    #
    #     eem1 = ee.predict_m1()
    #     eem2 = ee.predict_m2()
    #     eem3 = ee.predict_m3()
    #
    #     if config["volume"] == 'Insp':
    #         weights = [eim1, eim2, eim3]
    #         sf.export_to_cm(pmesh, weights, name=pfts['subject'], lung='L', show_mesh=False)
    #         sf.export_to_cm(pmesh, weights, name=pfts['subject'], lung='R', show_mesh=False)
    #     elif config["volume"] == 'Expn':
    #         weights = [eem1, eem2, eem3]
    #         sf.export_to_cm(pmesh, weights, name=pfts['subject'], lung='L', show_mesh=False)
    #         sf.export_to_cm(pmesh, weights, name=pfts['subject'], lung='R', show_mesh=False)

    # scores, mahalanobis_distance = _get_score(sf, aligned_mesh_names)

    return sf, pmesh


if __name__ == "__main__":
    sf, pmesh = main()
