import os
import json
from src.sfeal.core import SSM, SSM_Mesh
import matplotlib.pyplot as plt

from src.sfeal.core import SSM, SSM_Mesh
from src.sfeal.visualise import FIGURE

import numpy as np

from src.mlr import EiMLR, EeMLR
import pickle
import subprocess

CONFIGipf = os.path.join(os.path.dirname(__file__), '../', 'resources/ipfconfig.cnfg')
subject_list = []
with open(CONFIGipf) as config_file:
    cfg_ipf = json.load(config_file)

CONFIG = os.path.join(os.path.dirname(__file__), '../', 'resources/configs.cnfg')


class Main(object):

    def __init__(self, number_of_modes=None):
        self._cfg = None
        self._subjects = list()
        self._ref_subject = None
        self._aligned_mesh_objects = list()
        self._aligned_subjects = list()

        self._sfeal = SSM()
        self._mesh = SSM_Mesh()
        # self._fig = FIGURE()

        self._pmesh = None

        self.__read_config()
        self.__get_subjects()
        self.__align()

        if number_of_modes is None:
            self._number_of_modes = len(self._subjects) - 1
        else:
            self._number_of_modes = number_of_modes

    def __read_config(self):
        with open(CONFIG) as config_file:
            self._cfg = json.load(config_file)

    def __get_subjects(self):
        init_path = os.path.join(self._cfg["root"], self._cfg["study"])
        for subject in range(len(self._cfg["subjects"])):
            path_to_subject = os.path.join(init_path, self._cfg["subjects"][subject])
            path_to_dir = os.path.join(path_to_subject,
                                       self._cfg["protocol"],
                                       self._cfg["path"],
                                       self._cfg["raw"])
            path_to_file = os.path.join(path_to_dir, self._cfg["lung"] + "_fitted.mesh")
            if os.path.exists(path_to_file):
                if self._cfg["subjects"][subject] == "AGING025":
                    self._ref_subject = path_to_file
                self._subjects.append(path_to_file)
                # self.__add_mesh(path_to_file)

    def __add_mesh(self, mesh_file):
        self._sfeal.add_mesh(mesh_file)

    def __align(self, scaling=True):
        meshes = self._subjects
        reference_mesh = self._ref_subject
        print(reference_mesh)

        for mesh in meshes:
            d, Z, tform, m, aligned_mesh_path = self._mesh.align_mesh(reference_mesh, mesh, scaling=scaling, reflection='best')
            self._aligned_mesh_objects.append(m)
            self.__add_mesh(aligned_mesh_path)
            self._aligned_subjects.append(aligned_mesh_path)


    def __get_score(self):
        subject_names = sorted(self._aligned_subjects)
        m_distance = list()
        score_array = np.chararray((len(subject_names), self._sfeal.get_number_of_modes() + 1), itemsize=25, unicode=True)
        for i in range(len(subject_names)):
            score, ratio = self._sfeal.get_score(subject_names[i])
            mah_distance = self._sfeal.get_mahalanobis()
            m_distance.append(mah_distance)
            # print (subject_names)
            score_array[i][0] = subject_names[i].split('/')[6]
            for j in range(self._sfeal.num_modes):
                score_array[i][j + 1] = score['MODE   ' + str(j + 1) + ' SCORE']

        np.savetxt('aging_scoresScaled.csv', score_array, delimiter=',', fmt='%s')
        return m_distance

    def run(self):
        self._pmesh, _ = self._sfeal.pca_train(num_modes=self._number_of_modes)
        self._sfeal.save_mesh_id()
        self.__get_score()
    def export(self, weights, lung_side = 'L'):
        sm = self._sfeal
        pmesh = self._pmesh
        score = weights
        sm.export_to_cm(pmesh, score, lung=lung_side,show_mesh=True)

    def project(self, filename):
        sm = self._sfeal
        score, ratio = sm.project_new_mesh(filename)
        return score, ratio

    def generatemesh(self, filename1,lungS = 'L'):
        smesh = self._mesh
        smesh.generate_mesh(filename1,lung=lungS)

    def align_ipf(self, subs, scaling=True):
        sm = SSM()
        aligned_meshes = []
        meshes = subs
        reference_mesh = self._ref_subject
        print(reference_mesh)

        for mesh in meshes:
            d, Z, tform, m, aligned_mesh_path = self._mesh.align_mesh_ipf(reference_mesh, mesh, scaling=scaling,
                                                                      reflection='best')
            sm.add_mesh(aligned_mesh_path)
            aligned_meshes.append(aligned_mesh_path)
        return aligned_meshes

    def visual(self,mode,s1,s2,fiss):
        pmesh = self._pmesh
        fg = self._fig
        fg.spectrum(pmesh=pmesh,mode=mode,s1=s1,s2=s2,fissure=fiss)
        plt.show


def get_subjects(cfg_ipf,mesh):
    input_folder = 'sfeal/useful_files'
    init_path = os.path.join(cfg_ipf["root"], cfg_ipf["study"], cfg_ipf["mesh"])
    ip2py_perl = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'ip2py_'+cfg_ipf["lung"]+'.pl')

    for subject in range(len(cfg_ipf["subjects"])):
        subject_ID = cfg_ipf["subjects"][subject]
        path_to_subject = os.path.join(init_path, "IPF"+cfg_ipf["subjects"][subject])
        for date in range(len(cfg_ipf["subject_data_"+str(cfg_ipf["subjects"][subject])])):
            path_to_sub_date = os.path.join(path_to_subject,
                                            cfg_ipf["subject_data_"+subject_ID][date])

            path_to_file = os.path.join(path_to_sub_date, cfg_ipf["lung"] + "_fitted.ipnode")
            save_path = os.path.join(cfg_ipf["root"], cfg_ipf["study"], cfg_ipf["save"],  "IPF"+cfg_ipf["subjects"][subject],
                                     cfg_ipf["subject_data_"+subject_ID][date])
            path_to_save = os.path.join(save_path, cfg_ipf["lung"] + "_fitted.ip2py")
            if os.path.exists(path_to_file):
                print("file exists")

            if not os.path.exists(save_path):
                os.makedirs(save_path)


            # #subprocess to convert ipnode files into ip2py files
            # #convert both left and right lungs (by changing config files) prior to execution of next if loop
            # #after conversion, comment the subprocess function and uncomment next if loop for merging into one file
            # subprocess.Popen(["perl", ip2py_perl, path_to_file, path_to_save])

            # merge left and right ip2py files into Lung_fitted.ip2py
            if os.path.exists(os.path.join(save_path, "Right_fitted.ip2py")) and os.path.exists(os.path.join(save_path, "Left_fitted.ip2py")):
                print("in")
                file1 = os.path.join(save_path, "Right_fitted.ip2py")
                file2 = os.path.join(save_path, "Left_fitted.ip2py")
                filenames = [file1,file2]
                with open(os.path.join(save_path, "Lung_fitted.ip2py"), "w") as outfile:
                    for fname in filenames:
                        with open(fname) as infile:
                            outfile.write(infile.read())

                #generate morphic mesh using above merged file
                if os.path.exists(os.path.join(save_path, "Lung_fitted.ip2py")):
                    print("begin generation", save_path)
                    mesh.generate_mesh(save_path,lung='LR',save=True)
                    subject_list.append(os.path.join(save_path, "Lung_fitted.mesh"))
                    print("file exists!!!!")
    return subject_list

def projecMesh(sub_list,smesh):
    ###
    score_list = []
    for fname in sub_list:
        temp = fname.split('/')
        temp1 = str(temp[-4]+'_'+temp[-3])
        print(fname)
        score, ratio = main.project(fname)
        score_list.append((temp1,score['MODE_1 SCORE'],score['MODE_2 SCORE'],score['MODE_3 SCORE'],score['MODE_4 SCORE']))
    return np.array(score_list),ratio

def alignMesh (sub_list):
    aligned = main.align_ipf(sub_list)
    return aligned



# age, fvc, dlco, bmi, rvtlc, tlc
ei = EiMLR(20, 3.45, 30, 20, 27.25, 6.57)

m1 = ei.predict_m1()
m2 = ei.predict_m2()
m3 = ei.predict_m3()

print(m1,m2,m3)

if __name__ == '__main__':
    main = Main(number_of_modes=4)
    main.run()
    # with open('pca_data.pkl', 'wb') as output:
    #     pickle.dump(main, output, 0)
    print("######")
    weights = [0, 0, 0, 2.5]
    print('exporting')
    main.export(weights, 'L')
    print('done L')
    main.export(weights, 'R')
    print('done R')
    print("weights =", weights)
    print('done')
    # main.visual(1,-3,3,fiss=True)

    # main.generatemesh('/hpc/jjoh182/PythonProjectMaster/mahyar-sfeal/Ref_meshes/','LR')
    # score,ratio = main.project('/hpc/jjoh182/PythonProjectMaster/mahyar-sfeal/Aging_mesh/Lung_fitted.mesh') #file link to the py mesh of IPF subject
    # print(score,ratio)
             
mesh = SSM_Mesh()
smesh = SSM()

sub_list = get_subjects(cfg_ipf,mesh)
print(sub_list)

aligned_meshes = alignMesh(sub_list)

result,ratio = projecMesh(aligned_meshes,smesh)
np.savetxt('IPF_scoresAlignedScaled.csv', result, delimiter=',', fmt='%s')
print(result)
print('ratio is',ratio)
# score,ratio = main.project('/hpc/jjoh182/PythonProjectMaster/mahyar-sfeal/Aging_mesh/Lung_fitted.mesh')
