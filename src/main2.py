import os
import json

from src.sfeal.core import SSM, SSM_Mesh

import numpy as np

from src.mlr import EiMLR, EeMLR
import pickle

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

        self._pmesh = None

        self.subject_tform_dic = {}

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
        # self._ref_subject = os.path.join(init_path,"reference_mesh",
        #                                self._cfg["protocol"],
        #                                self._cfg["path"],
        #                                self._cfg["raw"],self._cfg["lung"] + "_fitted.mesh")


    def __add_mesh(self, mesh_file):
        self._sfeal.add_mesh(mesh_file)

    def __align(self, scaling=False):
        meshes = self._subjects
        reference_mesh = self._ref_subject

        for mesh in meshes:
            d, Z, tform, m, aligned_mesh_path = self._mesh.align_mesh(reference_mesh, mesh, scaling=scaling, reflection='best')
            self.subject_tform_dic[aligned_mesh_path] = tform
            self._aligned_mesh_objects.append(m)
            self.__add_mesh(aligned_mesh_path)
            self._aligned_subjects.append(aligned_mesh_path)

    def __get_score(self):
        subject_names = sorted(self._aligned_subjects)
        m_distance = list()
        score_array = np.chararray((len(subject_names), self._sfeal.get_number_of_modes() + 1), itemsize=25, unicode=True)
        ratio_array = np.chararray((len(subject_names), self._sfeal.get_number_of_modes() + 1), itemsize=25,
                                   unicode=True)
        for i in range(len(subject_names)):
            score, ratio = self._sfeal.get_score(subject_names[i])
            mah_distance = self._sfeal.get_mahalanobis()
            m_distance.append(mah_distance)
            # print (subject_names)
            score_array[i][0] = subject_names[i].split('/')[6]
            ratio_array[i][0] = subject_names[i].split('/')[6]
            for j in range(self._sfeal.num_modes):
                score_array[i][j + 1] = score['MODE   ' + str(j + 1) + ' SCORE']
                ratio_array[i][j + 1] = ratio['MODE   ' + str(j + 1) + ' RATIO']


        np.savetxt('aging_scores.csv', score_array, delimiter=',', fmt='%s')
        np.savetxt('aging_ratios.csv', ratio_array, delimiter=',', fmt='%s')
        return m_distance

    def run(self):
        self._pmesh, _ = self._sfeal.pca_train(num_modes=self._number_of_modes)
        # self._pmesh, _ = self._sfeal.pca_train(num_modes=1)
        self._sfeal.save_mesh_id()
        self.__get_score()


    def export(self, weights, lung_side = 'L'):
        sm = self._sfeal
        pmesh = self._pmesh
        score = weights
        sm.export_to_cm(pmesh, score, lung=lung_side,show_mesh=True)

    def export_special(self, weights,tform, lung_side = 'L'):
        sm = self._sfeal
        pmesh = self._pmesh
        score = weights
        sm.export_special(pmesh=pmesh, weights=score,tform=tform, lung=lung_side,show_mesh=True)


    def __project(self, filename):
        sm = self._sfeal
        score, ratio = sm.project_new_mesh(filename)
        return score, ratio

    def generatemesh(self, filename1,lungS = 'L'):
        smesh = self._mesh
        smesh.generate_mesh(filename1,lung=lungS)

    def projection_pipeline(self, mesh_dic, mesh_key, scaling):
        reference_mesh = self._ref_subject
        # d, Z, tform, m, aligned_mesh_path = self._mesh.align_mesh(reference_mesh, mesh, scaling=scaling, reflection='best')
        sm = self._sfeal
        score, ratio = self.__project(mesh_key)
        weight = list([float(score.get(k)) for k in ['MODE_1 SCORE','MODE_2 SCORE','MODE_3 SCORE']])
        # weight = list([float(score.get(k)) for k in ['MODE_1 SCORE']])
        # pmesh = self._pmesh
        tform = mesh_dic[mesh_key]

        main.export_special(weights=weight,lung_side='L',tform = tform)
        main.export_special(weights=weight,lung_side='R',tform = tform)
        from src.sfeal.morphic.mesher import Mesh
        # mesh_org = Mesh(aligned_mesh_path)
        # mesh_copy = Mesh(aligned_mesh_path)
        # projected_mesh = Mesh(mesh_path)
        # mesh_nodes = projected_mesh.get_nodes()
        #
        # #############
        # inv_rot = np.linalg.inv(tform['rotation'])
        # mesh_trans =(-1*tform['translation'])+mesh_nodes
        # mesh_trans = mesh_trans@inv_rot
        # Zlist = mesh_trans
        # for num, object in enumerate(mesh_copy.nodes):
        #     pmesh.nodes[object.id].values[:, 0] = Zlist[num]
        # main.export(weights=weight,mesh=mesh_copy,lung_side='R')


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
    # print("######")
    # weights = [0,0,0]
    # print('exporting')
    # main.export(weights, 'L')
    # print('done L')
    # main.export(weights, 'R')
    # print('done R')
    # print("weights =", weights)
    # print('done')

    # main.generatemesh('/hpc/jjoh182/PythonProjectMaster/mahyar-sfeal/Ref_meshes/','LR')
    # score,ratio = main.project('/hpc/jjoh182/PythonProjectMaster/'
    #              'mahyar-sfeal/Aging_mesh/Lung_fitted.mesh') #file link to the py mesh of IPF subject
    # print(score,ratio)

    # main.projection_pipeline('/hpc/jjoh182/PythonProjectMaster/mahyar-sfeal/Aging_mesh/Lung_fitted.mesh',scaling=False)
    main.projection_pipeline(main.subject_tform_dic, '/hpc/jjoh182/lung/Data/HLA_HA/AGING001/Insp/original/cmiss/morphic_aligned/Lung_fitted.mesh',scaling=False)


