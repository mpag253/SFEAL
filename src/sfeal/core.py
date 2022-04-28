import numpy as np
from src.sfeal.morphic.mesher import Mesh, DepNode, StdNode, PCANode
import src.ipnode2exnode
import pickle

class SSM(object):

    def __init__(self):

        self.X = list()
        self.input_mesh = None
        self.groups = None
        self.mesh = None
        self.pcamesh = None
        self.pmesh = None
        self.score_0 = None
        self.score_1 = list()
        self.z_score = dict()
        self.ratio = dict()
        self.fname = None
        self.mean = None
        self.score_z = None
        self.mah_distance = None
        self.SD = None
        self.nodes = None
        self.new_data = list()
        self.weights = list()
        self.bodies = None
        self.dataset = dict()
        self.mesh_names = list()

    def get_number_of_modes(self):
        return self.num_modes

    def add_mesh(self, m):
        mesh = Mesh(str(m))
        if self.input_mesh is None:
            self.input_mesh = mesh
        if isinstance(mesh, str):
            mesh = Mesh(mesh)
        x = list()
        if self.groups is None:
            for node in mesh.nodes:
                if not isinstance(node, DepNode):
                    x.extend(node.values.flatten().tolist())

        else:
            for node in mesh.nodes:
                if node.in_group(self.groups):
                    x.extend(node.values.flatten().tolist())

        self.mesh_names.append(str(m))
        self.X.append(x)

    def save_mesh_id(self):
        import pickle

        if len(self.mesh_names) == 0:
            raise ValueError("Data is empty! Have you generated the meshes?!")

        for i in range(len(self.mesh_names)):
            self.dataset.update({self.mesh_names[i]: i})

        fname = 'mesh_id.pkl'
        with open(fname, 'wb') as f:
            print(pickle.dump(self.dataset, f))

    def pca_train(self, pca_id, num_modes=2):
        from sklearn import decomposition
        import joblib

        self.X = np.array(self.X)
        #print("Shape 'self.X': ", np.shape(self.X))
        dumpfile = open("latest_pca_data_array.pkl", "wb")
        pickle.dump(self.X, dumpfile)
        dumpfile.close()
        self.num_modes = num_modes
        self.pca = decomposition.PCA(n_components=num_modes)
        self.pca.fit(self.X)
        self.mean = self.pca.mean_
        self.components = self.pca.components_.T
        self.variance = self.pca.explained_variance_
        self.generate_mesh()

        joblib.dump(self.pca, 'output/pca_model_'+pca_id+'.sfeal')
        # joblib.dump(self.mesh, 'lung_pca.model')
        return self.mesh, self.X

    def generate_mesh(self):
        self.mesh = Mesh()
        weights = np.zeros(self.num_modes + 1)
        weights[0] = 1.0
        self.mesh.add_stdnode('weights', weights)
        variance = np.zeros(self.num_modes + 1)
        variance[0] = 1.0
        variance[1:] = np.sqrt(self.variance)
        self.mesh.add_stdnode('variance', variance)
        idx = 0
        if self.groups is None:
            for node in self.input_mesh.nodes:
                nsize = node.values.size
                x = self.get_pca_node_values(node, idx)
                self.mesh.add_pcanode(node.id, x, 'weights', 'variance', group='pca')
                idx += nsize

        else:
            for node in self.input_mesh.nodes:
                nsize = node.values.size
                if node.in_group(self.groups):
                    x = self.get_pca_node_values(node, idx)
                    self.mesh.add_pcanode(node.id, x, 'weights', 'variance', group='pca')
                    idx += nsize
                else:
                    if isinstance(node, StdNode):
                        self.mesh.add_stdnode(node.id, node.values)
                    elif isinstance(node, DepNode):
                        self.mesh.add_depnode(node.id, node.element, node.node, shape=node.shape, scale=node.scale)
                    if isinstance(node, PCANode):
                        raise Exception('Not implemented')

        for element in self.input_mesh.elements:
            self.mesh.add_element(element.id, element.basis, element.node_ids)
        self.mesh.generate()

    def save_dataset(self, filename='mesh_dataset.data'):
        import pickle
        import os

        if len(self.X) == 0:
            raise ValueError("Data is empty! Have you generated the meshes?!")

        fname = os.path.join(os.path.dirname(__file__), filename)
        with open(fname, 'wb') as f:
            print(pickle.dump(self.X, f))

    def get_pca_node_values(self, node, idx):
        nsize = node.values.size
        if len(node.shape) == 1:
            pca_node_shape = (node.shape[0], 1, self.num_modes)
            x = np.zeros((node.shape[0], 1, self.num_modes + 1))
            x[:, 0, 0] = self.mean[idx:idx + nsize].reshape(node.shape)
            x[:, :, 1:] = self.components[idx:idx + nsize, :].reshape(pca_node_shape)
            return x
        if len(node.shape) == 2:
            pca_node_shape = (node.shape[0], node.shape[1], self.num_modes)
            x = np.zeros((node.shape[0], node.shape[1], self.num_modes + 1))
            x[:, :, 0] = self.mean[idx:idx + nsize].reshape(node.shape)
            x[:, :, 1:] = self.components[idx:idx + nsize, :].reshape(pca_node_shape)
            return x
        print('Cannot reshape this node when generating pca mesh')

    def _get_computations(self, mesh_file, pca_id):
        import joblib

        if not self.new_data:
            pass
        else:
            self.new_data = list()

        def search(values, searchFor):
            for k in values:
                if searchFor in k:
                    return k
            return None

        subject_name = search(self.dataset, mesh_file)
        # else:
        #     mah = True

        print('\n\t=========================================\n')
        #print('\t   Please wait... \n')

        size = self.X.shape[1] // 12
        total_subjects = len(self.dataset)
        if type(self.X) is not np.ndarray:
            self.X = np.array(self.X)
        X = self.X.reshape((total_subjects, size * 12))
        
        pca = joblib.load('output/pca_model_'+pca_id+'.sfeal')
        pca_mean = pca.mean_
        pca_mean = pca_mean.reshape((1, size * 12))
        pca_components = pca.components_.T
        pca_variance = pca.explained_variance_
        pca_explained_variance = pca.explained_variance_ratio_

        self.ratio = {}
        self.ratio = {
            'MODE {:3d} RATIO'.format(m + 1): '{:.4f}'.format(float(pca_explained_variance[m]))
            for m in range(len(pca_explained_variance))
        }

        count = len(pca_variance)
        mode_count = list()
        for i in range(len(pca_variance)):
            mode_count.append(i + 1)

        print('\t   Total modes of variation = {}'.format(count))
        print('\t   PROJECTING SUBJECT:')
        print('\t   {}'.format(mesh_file))

        mode_scores = list()
        for j in range(len(self.dataset)):
            subject = X[j] - pca_mean
            score = np.dot(subject, pca_components)
            mode_scores.append(score[0][0:count])

        self.SD = np.std(mode_scores, axis=0)
        self.mean = np.average(mode_scores, axis=0)
        number = self.dataset[subject_name]
        subject_0 = X[number] - pca_mean
        self.score_0 = np.dot(subject_0, pca_components)
        mah_distances = self.mahalanobis(self.score_0, pca_variance)
        self.mah_distance = np.mean(mah_distances)
        self.score_0 = self.score_0[0][0:count]
        self.score_1 = self.convert_scores(self.score_0, self.SD, self.mean)
        self.score_z = {
            'MODE {:3d} SCORE'.format(m + 1): '{:.2f}'.format(float(self.score_1[m]))
            for m in
            range(len(self.score_1))
        }
        print('\n\t=========================================\n')
        return None

    def calculate_score(self, mesh_file):
        import warnings
        warnings.warn("'calculate_score()' is depreciating in new versions of SFEAL. For future please use "
                      "'get_score()' method.", DeprecationWarning)

        self._get_computations(mesh_file)
        return self.score_z, self.ratio

    def mahalanobis(self, score, eigenvalues):
        mah_calc = 0
        for i in range(len(score)):
            calc = (score[i] * score[i]) / eigenvalues[i]
            mah_calc = mah_calc + calc
        return np.sqrt(mah_calc)

    def get_score(self, mesh_file, pca_id):
        self._get_computations(mesh_file, pca_id)
        return self.score_z, self.ratio

    def get_mahalanobis(self):
        return self.mah_distance

    def convert_scores(self, scores, SD, mean):
        self.score_1 = list()
        for i in range(len(scores)):
            self.score_1.append((scores[i] - mean[i]) / SD[i])
        return self.score_1

    def export_special(self, pmesh, weights,tform, name='default', lung='L', show_mesh=False):
        if not self.weights:
            pass
        else:
            self.weights = list()
        import os

        if lung == 'R':
            self.lung = 'Right'
        elif lung == 'L':
            self.lung = 'Left'
        else:
            raise Exception("'lung' argument can ONLY be L OR R!")

        self.weights = weights
        self.pmesh = pmesh
        self.pmesh.nodes['weights'].values[1:] = 0  # reset weights to zero
        self.pmesh.nodes['weights'].values[0] = 1  # adding average

        for num_mode in range(len(self.weights)):
            self.pmesh.nodes['weights'].values[num_mode + 1] = self.weights[num_mode]
            self.pmesh.update_pca_nodes()

        inv_rot = np.linalg.inv(tform['rotation'])
        # mesh_trans =(-1*tform['translation'])+mesh_nodes
        # mesh_trans = mesh_trans@inv_rot
        # Zlist = mesh_trans

        # # saving
        import os
        import pandas as pd
        import subprocess
        from src.sfeal.useful_files import nodes
        save_temp_file = "dummyfile.csv"
        # if self.output is not None:
        #     self.output = None
        if self.nodes is not None:
            self.nodes = None

        self.nodes = nodes.Nodes()

        root = '/hpc/jjoh182/PythonProjectMaster/mahyar-sfeal/src'

        output_dir = 'output/export_to_cm/%s/Insp' % name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        input_folder = 'useful_files'

        path_to_export_mesh = output_dir
        temp_file = '%s_reconstructed_temp.csv' % self.lung
        # output_file = '%s_reconstructed' % self.lung
        output_file = 'fitted%s' % self.lung
        save_temp_file = output_dir + '/%s' % temp_file
        save_output_file = output_dir + '/%s' % output_file
        # ipnode_file = '%s_reconstructed' % self.lung
        ipnode_file = 'fitted%s' % self.lung
        path_to_ipnode_file = os.path.join(root, '%s/%s' % (output_dir, ipnode_file))

        path_to_com_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'ipnode2exnode.com')
        path_to_cmgui_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'cmgui.com')
        ip2ex_perl = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'ipnode2exnode.pl')
        ip2ex_cm = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'ipnode2exnode')
        cmgui_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'cmgui')
        param_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', '3d_fitting')
        versions_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'versions')
        base_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'BiCubic_Surface_Unit')

        # root = '/hpc/jjoh182/PythonProjectMaster/mahyar-sfeal/src'
        # # nodes = self.nodes.set_nodes(lung='right')
        # # for node_number in nodes:
        # #     node = self.pmesh.nodes[node_number]
        # #     node_values = node.values
        # #     with open(save_temp_file, 'a') as f:
        # #         np.savetxt(f, node_values)
        # # self.output = 'morphic_aligned'
        #
        # output_dir = 'output/export_to_cm/%s/Insp' % name
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        #
        # input_folder = 'useful_files'
        #
        # path_to_export_mesh = output_dir
        # temp_file = '%s_reconstructed_temp.csv' % self.lung
        # # output_file = '%s_reconstructed' % self.lung
        # output_file = 'fitted%s' % self.lung
        # save_temp_file = output_dir + '/%s' % temp_file
        # save_output_file = output_dir + '/%s' % output_file
        #
        # ipnode_file = 'fitted%s' % self.lung
        # path_to_ipnode_file = os.path.join(root, '%s/%s' % (output_dir, ipnode_file))
        #
        # path_to_com_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'ipnode2exnode.com')
        # path_to_cmgui_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'cmgui.com')
        # ip2ex_perl = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'ipnode2exnode.pl')
        # ip2ex_cm = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'ipnode2exnode')
        # cmgui_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'cmgui')
        # param_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', '3d_fitting')
        # versions_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'versions')
        # base_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'BiCubic_Surface_Unit')


        # mesh_output = os.path.normpath(output_dir + os.sep + os.pardir)
        # mesh_output = os.path.normpath(mesh_output + os.sep + os.pardir)
        # #
        # # mesh_output = os.path.join(mesh_output, self.output)
        # #
        # if not os.path.exists(mesh_output):
        #     os.makedirs(mesh_output)

        if self.lung == 'Right':
            node_file = 'nodes_%s.csv' % self.lung
            input_file = os.path.join(os.path.dirname(__file__), input_folder, node_file)
            elem_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'templateRight')

            nodes = self.nodes.set_nodes(lung='right')
            for node_number in nodes:
                node = self.pmesh.nodes[node_number]
                node_values = node.values
                with open(save_temp_file, 'a') as f:
                    np.savetxt(f, node_values)

            a = pd.read_csv(input_file)
            b = pd.read_csv(save_temp_file, delimiter=' ')
            barray = np.array(b)
            b_ex = barray[:,0:3]
            trans_nodes =(-1*tform['translation'])+b_ex
            trans_nodes = trans_nodes@inv_rot
            bnew = np.zeros(barray.shape)
            bnew[:,0:3]=trans_nodes
            b_df = pd.DataFrame(bnew)
            b_df.reset_index(drop=True)

            result = pd.concat([a, b_df], axis=1,ignore_index=True)
            result.to_csv('%s.csv' % save_output_file, sep=' ', index=False)
            os.remove(save_temp_file)

            py2ip_right_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'py2ip_right.pl')

            subprocess.Popen(["perl", py2ip_right_file, "%s.csv" % save_output_file, "%s.ipnode" % path_to_ipnode_file])

            with open(path_to_com_file, 'w') as comfile:
                comfile.write(" set echo on;\n")
                comfile.write(" fem def param;r;{0}".format("%s;\n" % param_file))
                comfile.write(" fem def coor;r;{0}".format("%s;\n" % versions_file))
                comfile.write(" fem def base;r;{0}".format("%s;\n" % base_file))
                comfile.write(" fem def node;r;{0}".format("%s;\n" % path_to_ipnode_file))
                comfile.write(" fem def elem;r;{0}".format("%s;\n" % elem_file))
                comfile.write(" fem export node;{0} as fitted{1};\n".format("%s" % path_to_ipnode_file, self.lung))
                comfile.write(" fem export elem;{0} as fitted{1};\n".format("%s" % path_to_ipnode_file, self.lung))
                comfile.write(" fem def node;w;{0}".format("%s;\n" % path_to_ipnode_file))
                comfile.write(" fem quit;\n")

            if show_mesh:
                with open(path_to_cmgui_file, 'w') as comfile: #wb to w to avoid bytes-like error
                    comfile.write(" gfx read node {0}".format("'%s';\n" % path_to_ipnode_file))
                    comfile.write(" gfx read elem {0}".format("'%s';\n" % path_to_ipnode_file))
                    comfile.write(" gfx cre egroup fissure;\n")
                    comfile.write(" gfx mod egroup fissure add 51..62;\n")
                    comfile.write(
                        " gfx mod g_e {0} general clear circle_discretization 6 default_coordinate coordinates; element_discretization '12*12*12' native_discretization none;\n".format(
                            "'%s'" % self.lung))
                    comfile.write(
                        " gfx mod g_e {0} lines coordinate coordinates select_on material green selected_material default_selected;\n".format(
                            "'%s'" % self.lung))
                    comfile.write(
                        " gfx mod g_e fissure general clear circle_discretization 6 default_coordinate coordinates; element_discretization '12*12*12' native_discretization none;\n")
                    comfile.write(" gfx mod g_e fissure surfaces material tissue;\n")
                    comfile.write(" gfx edit scene;\n")
                    comfile.write(" gfx cre win;\n")
                show_cmgui = 'show'
                subprocess.Popen(["perl", ip2ex_perl, "%s" % ip2ex_cm])
                # subprocess.Popen(["perl", ip2ex_perl, "%s" % ip2ex_cm, "%s" % cmgui_file, "%s" % show_cmgui])

            else:
                show_cmgui = 'no'
                # subprocess.call(["perl", ip2ex_perl, "%s" % ip2ex_cm, "%s" % cmgui_file, "%s" % show_cmgui],
                #                 )

            print("\n\t=========================================\n")
            print("\t   ALL MESH FILES EXPORTED TO:")
            print("\n\t\t   {0} ".format(path_to_export_mesh))
            print("\n\t=========================================\n")

        elif self.lung == 'Left':

            node_file = 'nodes_%s.csv' % self.lung
            input_file = os.path.join(os.path.dirname(__file__), input_folder, node_file)
            elem_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'templateLeft')

            nodes = self.nodes.set_nodes(lung='left')
            for node_number in nodes:
                node = self.pmesh.nodes[node_number]
                node_values = node.values
                with open(save_temp_file, 'a') as f:
                    np.savetxt(f, node_values)
            print(input_file)
            a = pd.read_csv(input_file)
            b = pd.read_csv(save_temp_file, delimiter=' ')
            barray = np.array(b)
            b_ex = barray[:,0:3]
            trans_nodes =(-1*tform['translation'])+b_ex
            trans_nodes = trans_nodes@inv_rot
            bnew = np.zeros(barray.shape)
            bnew[:,0:3]=trans_nodes
            b_df = pd.DataFrame(bnew)

            result = pd.concat([a, b_df], axis=1)
            result.to_csv('%s.csv' % save_output_file, sep=' ', index=False)
            os.remove(save_temp_file)

            py2ip_left_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'py2ip_left.pl')
            subprocess.Popen(["perl", py2ip_left_file, "%s.csv" % save_output_file, "%s.ipnode" % path_to_ipnode_file])

            with open(path_to_com_file, 'w') as comfile:
                comfile.write(" set echo on;\n")
                comfile.write(" fem def param;r;{0}".format("%s;\n" % param_file))
                comfile.write(" fem def coor;r;{0}".format("%s;\n" % versions_file))
                comfile.write(" fem def base;r;{0}".format("%s;\n" % base_file))
                comfile.write(" fem def node;r;{0}".format("%s;\n" % path_to_ipnode_file))
                comfile.write(" fem def elem;r;{0}".format("%s;\n" % elem_file))
                comfile.write(" fem export node;{0} as fitted{1};\n".format("%s" % path_to_ipnode_file, self.lung))
                comfile.write(" fem export elem;{0} as fitted{1};\n".format("%s" % path_to_ipnode_file, self.lung))
                comfile.write(" fem def node;w;{0}".format("%s;\n" % path_to_ipnode_file))
                comfile.write(" fem quit;\n")

            if show_mesh:
                with open(path_to_cmgui_file, 'w') as comfile:
                    comfile.write(" gfx read node {0}".format("'%s';\n" % path_to_ipnode_file))
                    comfile.write(" gfx read elem {0}".format("'%s';\n" % path_to_ipnode_file))
                    comfile.write(" gfx cre egroup fissure;\n")
                    comfile.write(" gfx mod egroup fissure add 111..118;\n")
                    comfile.write(
                        " gfx mod g_e {0} general clear circle_discretization 6 default_coordinate coordinates; element_discretization '12*12*12' native_discretization none;\n".format(
                            "'%s'" % self.lung))
                    comfile.write(
                        " gfx mod g_e {0} lines coordinate coordinates select_on material green selected_material default_selected;\n".format(
                            "'%s'" % self.lung))
                    comfile.write(
                        " gfx mod g_e fissure general clear circle_discretization 6 default_coordinate coordinates; element_discretization '12*12*12' native_discretization none;\n")
                    comfile.write(" gfx mod g_e fissure surfaces material tissue;\n")
                    comfile.write(" gfx edit scene;\n")
                    comfile.write(" gfx cre win;\n")
                show_cmgui = 'show'

                # src.ipnode2exnode.ipread(ip2ex_perl)
                print(ip2ex_perl, ip2ex_cm)
                subprocess.Popen(["perl", ip2ex_perl, "%s" % ip2ex_cm])

            else:
                show_cmgui = 'no'
                # subprocess.call(["perl", ip2ex_perl, "%s" % ip2ex_cm, "%s" % cmgui_file, "%s" % show_cmgui],
                #                 )

            print("\n\t=========================================\n")
            print("\t   ALL MESH FILES EXPORTED TO:")
            print("\t   {0} ".format(path_to_export_mesh))
            print("\n\t=========================================\n")

        return



    def export_to_cm(self, pmesh, weights, name='default', body='L', show_mesh=False):
        if self.weights is None:  # if not self.weights:
            pass
        else:
            self.weights = list()
        import os

        if body == 'R':
            self.body = 'Right'
        elif body == 'L':
            self.body = 'Left'
        elif body == 'T':
            self.body = 'Torso'
        else:
            raise Exception("'body' argument can ONLY be L, R, or T!")

        self.weights = weights
        self.pmesh = pmesh
        self.pmesh.nodes['weights'].values[1:] = 0  # reset weights to zero
        self.pmesh.nodes['weights'].values[0] = 1  # adding average

        for num_mode in range(len(self.weights)):
            self.pmesh.nodes['weights'].values[num_mode + 1] = self.weights[num_mode]
            self.pmesh.update_pca_nodes()

        # saving
        import os
        import pandas as pd
        import subprocess
        from src.sfeal.useful_files import nodes

        if self.nodes is not None:
            self.nodes = None

        self.nodes = nodes.Nodes()

        root = '/hpc/mpag253/Torso/SFEAL'

        output_dir = 'output/export_to_cm/%s' % name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        input_folder = 'useful_files'

        path_to_export_mesh = output_dir
        temp_file = '%s_reconstructed_temp.csv' % self.body
        output_file = 'fitted%s' % self.body
        save_temp_file = output_dir + '/%s' % temp_file
        save_output_file = output_dir + '/%s' % output_file
        ipnode_file = 'fitted%s' % self.body
        path_to_ipnode_file = os.path.join(root, '%s/%s' % (output_dir, ipnode_file))

        path_to_com_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'ipnode2exnode.com')
        path_to_cmgui_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'cmgui.com')
        ip2ex_perl = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'ipnode2exnode.pl')
        ip2ex_cm = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'ipnode2exnode')
        cmgui_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'cmgui')
        param_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', '3d_fitting')
        versions_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'versions')
        base_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'BiCubic_Surface_Unit')

        if self.body == 'Right':
            node_file = 'nodes_%s.csv' % self.body
            input_file = os.path.join(os.path.dirname(__file__), input_folder, node_file)
            elem_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'templateRight')

            nodes = self.nodes.set_nodes(bodies='right')
            for node_number in nodes:
                node = self.pmesh.nodes[node_number]
                node_values = node.values
                with open(save_temp_file, 'a') as f:
                    np.savetxt(f, node_values)

            a = pd.read_csv(input_file)
            b = pd.read_csv(save_temp_file, delimiter=' ')
            result = pd.concat([a, b], axis=1)
            result.to_csv('%s.csv' % save_output_file, sep=' ', index=False)
            os.remove(save_temp_file)
            py2ip_right_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'py2ip_right.pl')

            subprocess.Popen(["perl", py2ip_right_file, "%s.csv" % save_output_file, "%s.ipnode" % path_to_ipnode_file])

            with open(path_to_com_file, 'w') as comfile:
                comfile.write(" set echo on;\n")
                comfile.write(" fem def param;r;{0}".format("%s;\n" % param_file))
                comfile.write(" fem def coor;r;{0}".format("%s;\n" % versions_file))
                comfile.write(" fem def base;r;{0}".format("%s;\n" % base_file))
                comfile.write(" fem def node;r;{0}".format("%s;\n" % path_to_ipnode_file))
                comfile.write(" fem def elem;r;{0}".format("%s;\n" % elem_file))
                comfile.write(" fem export node;{0} as fitted{1};\n".format("%s" % path_to_ipnode_file, self.body))
                comfile.write(" fem export elem;{0} as fitted{1};\n".format("%s" % path_to_ipnode_file, self.body))
                comfile.write(" fem def node;w;{0}".format("%s;\n" % path_to_ipnode_file))
                comfile.write(" fem quit;\n")

            if show_mesh:
                with open(path_to_cmgui_file, 'w') as comfile: #wb to w to avoid bytes-like error
                    comfile.write(" gfx read node {0}".format("'%s';\n" % path_to_ipnode_file))
                    comfile.write(" gfx read elem {0}".format("'%s';\n" % path_to_ipnode_file))
                    comfile.write(" gfx cre egroup fissure;\n")
                    comfile.write(" gfx mod egroup fissure add 51..62;\n")
                    comfile.write(
                        " gfx mod g_e {0} general clear circle_discretization 6 default_coordinate coordinates; element_discretization '12*12*12' native_discretization none;\n".format(
                            "'%s'" % self.body))
                    comfile.write(
                        " gfx mod g_e {0} lines coordinate coordinates select_on material green selected_material default_selected;\n".format(
                            "'%s'" % self.body))
                    comfile.write(
                        " gfx mod g_e fissure general clear circle_discretization 6 default_coordinate coordinates; element_discretization '12*12*12' native_discretization none;\n")
                    comfile.write(" gfx mod g_e fissure surfaces material tissue;\n")
                    comfile.write(" gfx edit scene;\n")
                    comfile.write(" gfx cre win;\n")
                show_cmgui = 'show'
                subprocess.Popen(["perl", ip2ex_perl, "%s" % ip2ex_cm])
                # subprocess.Popen(["perl", ip2ex_perl, "%s" % ip2ex_cm, "%s" % cmgui_file, "%s" % show_cmgui])

            else:
                show_cmgui = 'no'
                # subprocess.call(["perl", ip2ex_perl, "%s" % ip2ex_cm, "%s" % cmgui_file, "%s" % show_cmgui],
                #                 )

            print("\n\t=========================================\n")
            print("\t   ALL MESH FILES EXPORTED TO:")
            print("\t   " + str(path_to_export_mesh))
            print("\n\t=========================================\n")

            # os.remove(save_output_file + '.csv')

        elif self.body == 'Left':

            node_file = 'nodes_%s.csv' % self.body
            input_file = os.path.join(os.path.dirname(__file__), input_folder, node_file)
            elem_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'templateLeft')

            nodes = self.nodes.set_nodes(bodies='left')
            for node_number in nodes:
                node = self.pmesh.nodes[node_number]
                node_values = node.values
                with open(save_temp_file, 'a') as f:
                    np.savetxt(f, node_values)
            print(input_file)
            a = pd.read_csv(input_file)
            b = pd.read_csv(save_temp_file, delimiter=' ')
            result = pd.concat([a, b], axis=1)
            result.to_csv('%s.csv' % save_output_file, sep=' ', index=False)
            os.remove(save_temp_file)

            py2ip_left_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'py2ip_left.pl')
            # print("inside export cm")
            # print(py2ip_left_file,save_output_file,path_to_ipnode_file)
            # print(["perl", py2ip_left_file, "%s.csv" % save_output_file, "%s.ipnode" % path_to_ipnode_file])
            subprocess.Popen(["perl", py2ip_left_file, "%s.csv" % save_output_file, "%s.ipnode" % path_to_ipnode_file])
            # print(path_to_com_file,path_to_ipnode_file)
            with open(path_to_com_file, 'w') as comfile:
                comfile.write(" set echo on;\n")
                comfile.write(" fem def param;r;{0}".format("%s;\n" % param_file))
                comfile.write(" fem def coor;r;{0}".format("%s;\n" % versions_file))
                comfile.write(" fem def base;r;{0}".format("%s;\n" % base_file))
                comfile.write(" fem def node;r;{0}".format("%s;\n" % path_to_ipnode_file))
                comfile.write(" fem def elem;r;{0}".format("%s;\n" % elem_file))
                comfile.write(" fem export node;{0} as fitted{1};\n".format("%s" % path_to_ipnode_file, self.body))
                comfile.write(" fem export elem;{0} as fitted{1};\n".format("%s" % path_to_ipnode_file, self.body))
                comfile.write(" fem def node;w;{0}".format("%s;\n" % path_to_ipnode_file))
                comfile.write(" fem quit;\n")

            if show_mesh:
                with open(path_to_cmgui_file, 'w') as comfile:
                    comfile.write(" gfx read node {0}".format("'%s';\n" % path_to_ipnode_file))
                    comfile.write(" gfx read elem {0}".format("'%s';\n" % path_to_ipnode_file))
                    comfile.write(" gfx cre egroup fissure;\n")
                    comfile.write(" gfx mod egroup fissure add 111..118;\n")
                    comfile.write(
                        " gfx mod g_e {0} general clear circle_discretization 6 default_coordinate coordinates; element_discretization '12*12*12' native_discretization none;\n".format(
                            "'%s'" % self.body))
                    comfile.write(
                        " gfx mod g_e {0} lines coordinate coordinates select_on material green selected_material default_selected;\n".format(
                            "'%s'" % self.body))
                    comfile.write(
                        " gfx mod g_e fissure general clear circle_discretization 6 default_coordinate coordinates; element_discretization '12*12*12' native_discretization none;\n")
                    comfile.write(" gfx mod g_e fissure surfaces material tissue;\n")
                    comfile.write(" gfx edit scene;\n")
                    comfile.write(" gfx cre win;\n")
                show_cmgui = 'show'

                # src.ipnode2exnode.ipread(ip2ex_perl)
                print(ip2ex_perl,ip2ex_cm)
                subprocess.Popen(["perl", ip2ex_perl, "%s" % ip2ex_cm])

            else:
                show_cmgui = 'no'
                # subprocess.call(["perl", ip2ex_perl, "%s" % ip2ex_cm, "%s" % cmgui_file, "%s" % show_cmgui],
                #                 )

            print("\n\t=========================================\n")
            print("\t   ALL MESH FILES EXPORTED TO:")
            print("\t   " + str(path_to_export_mesh))
            print("\n\t=========================================\n")

            # os.remove(save_output_file + '.csv')
            
        elif self.body == 'Torso':

            node_file = 'nodes_%s.csv' % self.body
            input_file = os.path.join(os.path.dirname(__file__), input_folder, node_file)
            elem_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'templateTorso')

            nodes = self.nodes.set_nodes(bodies='torso')
            for node_number in nodes:
                node = self.pmesh.nodes[node_number]
                node_values = node.values
                with open(save_temp_file, 'a') as f:
                    np.savetxt(f, node_values)
            #print(input_file)
            a = pd.read_csv(input_file)
            b = pd.read_csv(save_temp_file, delimiter=' ')
            result = pd.concat([a, b], axis=1)
            result.to_csv('%s.csv' % save_output_file, sep=' ', index=False)
            os.remove(save_temp_file)

            py2ip_torso_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'py2ip_torso.pl')

            subprocess.Popen(["perl", py2ip_torso_file, "%s.csv" % save_output_file, "%s.ipnode" % path_to_ipnode_file])
            # print(path_to_com_file,path_to_ipnode_file)
            with open(path_to_com_file, 'w') as comfile:
                comfile.write(" set echo on;\n")
                comfile.write(" fem def param;r;{0}".format("%s;\n" % param_file))
                comfile.write(" fem def coor;r;{0}".format("%s;\n" % versions_file))
                comfile.write(" fem def base;r;{0}".format("%s;\n" % base_file))
                comfile.write(" fem def node;r;{0}".format("%s;\n" % path_to_ipnode_file))
                comfile.write(" fem def elem;r;{0}".format("%s;\n" % elem_file))
                comfile.write(" fem export node;{0} as fitted{1};\n".format("%s" % path_to_ipnode_file, self.body))
                comfile.write(" fem export elem;{0} as fitted{1};\n".format("%s" % path_to_ipnode_file, self.body))
                comfile.write(" fem def node;w;{0}".format("%s;\n" % path_to_ipnode_file))
                comfile.write(" fem quit;\n")

            if show_mesh:
                with open(path_to_cmgui_file, 'w') as comfile:
                    comfile.write(" gfx read node {0}".format("'%s';\n" % path_to_ipnode_file))
                    comfile.write(" gfx read elem {0}".format("'%s';\n" % path_to_ipnode_file))
                    comfile.write(" gfx cre egroup fissure;\n")
                    comfile.write(" gfx mod egroup fissure add 111..118;\n")
                    comfile.write(
                        " gfx mod g_e {0} general clear circle_discretization 6 default_coordinate coordinates; element_discretization '12*12*12' native_discretization none;\n".format(
                            "'%s'" % self.body))
                    comfile.write(
                        " gfx mod g_e {0} lines coordinate coordinates select_on material green selected_material default_selected;\n".format(
                            "'%s'" % self.body))
                    comfile.write(
                        " gfx mod g_e fissure general clear circle_discretization 6 default_coordinate coordinates; element_discretization '12*12*12' native_discretization none;\n")
                    comfile.write(" gfx mod g_e fissure surfaces material tissue;\n")
                    comfile.write(" gfx edit scene;\n")
                    comfile.write(" gfx cre win;\n")
                show_cmgui = 'show'

                # src.ipnode2exnode.ipread(ip2ex_perl)
                print(ip2ex_perl,ip2ex_cm)
                subprocess.Popen(["perl", ip2ex_perl, "%s" % ip2ex_cm])

            else:
                show_cmgui = 'no'
                # subprocess.call(["perl", ip2ex_perl, "%s" % ip2ex_cm, "%s" % cmgui_file, "%s" % show_cmgui],
                #                 )

            print("\n\t=========================================\n")
            print("\t   ALL MESH FILES EXPORTED TO:")
            print("\t   " + str(path_to_export_mesh))
            print("\n\t=========================================\n")
            
            # os.remove(save_output_file + '.csv')
            
        return None
        
        
    def get_sampled_nodes(self, pmesh, weights):
        if self.weights is None:  # if not self.weights:
            pass
        else:
            self.weights = list()
        import os

        self.weights = weights
        self.pmesh = pmesh
        self.pmesh.nodes['weights'].values[1:] = 0  # reset weights to zero
        self.pmesh.nodes['weights'].values[0] = 1  # adding average

        for num_mode in range(len(self.weights)):
            self.pmesh.nodes['weights'].values[num_mode + 1] = self.weights[num_mode]
            self.pmesh.update_pca_nodes()
            
        from src.sfeal.useful_files import nodes
        if self.nodes is not None:
            self.nodes = None
        self.nodes = nodes.Nodes()
            
        nodes = self.nodes.set_nodes(bodies='lrt')
        node_data = np.zeros([len(nodes), 12])
        for i, node_number in enumerate(nodes):
            node = self.pmesh.nodes[node_number]
            node_values = node.values
            #print(node_number, "\n", node_values)
            node_data[i, :] = node_values.flatten()
            
        return node_data
        

    # def project_new_mesh(self, mesh_file_names, mesh_file):
    def project_new_mesh(self, mesh_file):
        import joblib
        import pickle

        print('\n\t=========================================\n')
        print('\t   Please wait... \n')

        size = self.X.shape[1] // 12
        total_subjects = len(self.X)
        if type(self.X) is not np.ndarray:
            self.X = np.array(self.X)
        X = self.X.reshape((total_subjects, size * 12))
        pca = joblib.load('lung_pca_model.sfeal')
        pca_mean = pca.mean_
        pca_mean = pca_mean.reshape((1, size * 12))
        pca_components = pca.components_.T
        pca_variance = pca.explained_variance_
        pca_explained_variance = pca.explained_variance_ratio_

        self.ratio = {}
        self.ratio = {'MODE_{} RATIO'.format(m + 1): '{:.2f}'.format(float(pca_explained_variance[m])) for m in
                      range(len(pca_explained_variance))}

        count = len(pca_variance)
        mode_count = list()
        for i in range(len(pca_variance)):
            mode_count.append(i + 1)

        print('\t   Total modes of variation = %d' % count)
        print('\t   Projecting Subject: %s' % mesh_file)

        mode_scores = list()
        for j in range(len(self.X)):
            subject = X[j] - pca_mean
            score = np.dot(subject, pca_components)
            mode_scores.append(score[0][0:count])

        if self.SD is not None:
            self.SD = None
        if self.mean is not None:
            self.mean = None

        self.SD = np.std(mode_scores, axis=0)
        self.mean = np.average(mode_scores, axis=0)

        project_mesh_path = mesh_file
        project_mesh = Mesh(project_mesh_path)

        y = list()
        for node in project_mesh.nodes:
            y.extend(node.values)
            Y = np.asarray(y)

        Y = Y.reshape((size * 12))
        subject_0 = Y - pca_mean

        if self.score_0 is not None:
            self.score_0 = None
        if self.score_z is not None:
            self.score_z = None

        self.score_0 = np.dot(subject_0, pca_components)
        self.score_0 = self.score_0[0][0:count]
        self.score_1 = list()
        self.score_1 = self.convert_scores(self.score_0, self.SD, self.mean)
        self.score_z = {}
        self.score_z = {'MODE_{} SCORE'.format(m + 1): '{:.2f}'.format(float(self.score_1[m])) for m in
                        range(len(self.score_1))}

        print('\n\t=========================================\n')
        return self.score_z, self.ratio


###################################################################################################
###################################################################################################
###################################################################################################


class SSM_Mesh(object):

    def __init__(self):
        self.bodies = None
        self.count = 0
        self.elements = None
        self.mesh = None
        self.output = None
        self.file_path = None

    def generate_mesh(self, file_path, bodies='L', save=True):
        """
        generate_mesh creates morphic meshes from finite element meshes
        built in cmiss. The input mesh which has already been converted
        into a matrix format is transformed into a morphic mesh using the
        node and element class within the useful_files module. These
        classes can be modified accordingly for other mesh topologies.

        Inputs:
        ------------
        file_path
            path where the .ipnode file is stored.

        file_name
            name of the .ipnode file (do not include the .ipnode itself).

        bodies
            Left = L | Right = R | Both = LR | Both + Torso = LRT

        save
            A boolean variable to save the mesh.

        Outputs:
        ------------
        None

        :param file_path
        :param file_name
        :param bodies
        :param save
        :return: None
        """
        import csv
        import os, sys
        from src.sfeal.useful_files import elements                      #########

        if bodies.lower() == 'l':
            self.bodies = 'Left'
        elif bodies.lower() == 'R' or bodies.lower() == 'r':
            self.bodies = 'Right'
        elif bodies.lower() == 'lr' or bodies.lower() == 'rl':
            self.bodies = 'Lung'
        elif bodies.lower() == 'lrt' or bodies.lower() == 'rlt':
            self.bodies = 'TorsoLung'

        if self.mesh is not None:
            self.mesh = None

        self.mesh = Mesh()
        data = {}

        if self.elements is not None:
            self.elements = None

        self.elements = elements.Elements()

        print('\n\t=========================================\n')
        print('\t   GENERATING MESH... \n')
        print('\t   PLEASE WAIT... \n')

        if self.file_path is not None:
            self.file_path = None

        self.file_path = file_path

        for filenum in os.listdir(self.file_path):
            filenum_path = os.path.join(self.file_path, filenum)
            if filenum_path == os.path.join(self.file_path, self.bodies + '_fitted.ip2py'):
                if os.path.isfile(filenum_path):
                    self.count += 1
                    with open(filenum_path, 'r') as csvfile:
                        data[filenum] = csv.reader(csvfile, delimiter=' ', quotechar='|')
                        for rowx in data[filenum]:
                            rowy = next(data[filenum])
                            rowz = next(data[filenum])
                            node = [[float(rowx[1]), float(rowx[2]), float(rowx[3]), float(rowx[4])],
                                    [float(rowy[1]), float(rowy[2]), float(rowy[3]), float(rowy[4])],
                                    [float(rowz[1]), float(rowz[2]), float(rowz[3]), float(rowz[4])]]
                            nd = self.mesh.add_stdnode(str(rowx[0]), node)

                        if self.bodies == 'Left':
                            elements = self.elements.set_elements(bodies='left')
                            for ii, elem in enumerate(elements):
                                self.mesh.add_element(ii + 1, ['H3', 'H3'], elem)
                        elif self.bodies == 'Right':
                            elements = self.elements.set_elements(bodies='right')
                            for ii, elem in enumerate(elements):
                                self.mesh.add_element(ii + 1, ['H3', 'H3'], elem)
                        elif self.bodies == 'Lung':
                            elements = self.elements.set_elements(bodies='lr')
                            for ii, elem in enumerate(elements):
                                self.mesh.add_element(ii + 1, ['H3', 'H3'], elem)
                        elif self.bodies == 'TorsoLung':                              #########
                            elements = self.elements.set_elements(bodies='lrt')         #########
                            for ii, elem in enumerate(elements):                        #########
                                self.mesh.add_element(ii + 1, ['H3', 'H3'], elem)       #########

                        self.mesh.generate()

                        if save:
                            mesh_output = os.path.normpath(filenum_path + os.sep + os.pardir)
                            self.mesh.save(mesh_output + '/' + self.bodies + '_fitted.mesh')

                            print('\t   MESH SAVED IN \n')
                            print('\t   %s DIRECTORY \n' % mesh_output)
        print('\n\t=========================================\n')

    def align_mesh(self, reference_mesh, mesh, bodies, scaling=True, reflection='best'):
        """
        align_mesh is a method that performes a Procrustes analysis which
        determines a linear transformation (translation, reflection, orthogonal rotation
        and scaling) of the nodes in mesh to best conform them to the nodes in reference_mesh,
        using the sum of squared errors as the goodness of fit criterion.

        Inputs:
        ------------
        reference_mesh, mesh
            meshes (as morphic meshes) of target and input coordinates. they must have equal
            numbers of  nodes (rows), but mesh may have fewer dimensions
            (columns) than reference_mesh.

        scaling
            if False, the scaling component of the transformation is forced
            to 1

        reflection
            if 'best' (default), the transformation solution may or may not
            include a reflection component, depending on which fits the data
            best. setting reflection to True or False forces a solution with
            reflection or no reflection respectively.

        Outputs
        ------------
        d
            the residual sum of squared errors, normalized according to a
            measure of the scale of reference_mesh, ((reference_mesh - reference_mesh.mean(0))**2).sum()

        Z
            the matrix of transformed Y-values

        tform
            a dict specifying the rotation, translation and scaling that
            maps X --> Y

        self.mesh
            Aligned mesh

        :param reference_mesh
        :param mesh
        :param scaling
        :param reflection
        :return: d, Z, tform, self.mesh
        """

        import os

        print('\n\t=========================================\n')
        print('\t   ALIGNING MESH... ')
        #print('\t   PLEASE WAIT... \n')

        if bodies.lower() == 'l':
            self.bodies = 'Left'
        elif bodies.lower() == 'R' or bodies.lower() == 'r':
            self.bodies = 'Right'
        elif bodies.lower() == 'lr' or bodies.lower() == 'rl':
            self.bodies = 'Lung'
        elif bodies.lower() == 'lrt' or bodies.lower() == 'rlt':
            self.bodies = 'TorsoLung'

        if self.mesh is not None:
            self.mesh = None

        r = Mesh(reference_mesh)
        self.mesh = Mesh(mesh)

        X = r.get_nodes()
        Y = self.mesh.get_nodes()

        n, m = X.shape
        ny, my = Y.shape

        muX = X.mean(0)
        muY = Y.mean(0)

        X0 = X - muX
        Y0 = Y - muY

        ssX = (X0 ** 2.).sum()
        ssY = (Y0 ** 2.).sum()
        
        """ centred Frobenius norm """
        normX = np.sqrt(ssX)
        normY = np.sqrt(ssY)
        
        """ scale to equal (unit) norm """
        X0 /= normX
        Y0 /= normY

        if my < m:
            Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

        """ optimum rotation matrix of Y """
        A = np.dot(X0.T, Y0)
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        V = Vt.T
        T = np.dot(V, U.T)

        if reflection != 'best':

            """ if the current solution use a reflection? """
            have_reflection = np.linalg.det(T) < 0

            """ if that's not what was specified, force another reflection """
            if reflection != have_reflection:
                V[:, -1] *= -1
                s[-1] *= -1
                T = np.dot(V, U.T)

        traceTA = s.sum()
        #print("ssX: \n", ssX)
        #print("traceTA: \n", traceTA)
        #print("normX: \n", normX)
        #print("normY: \n", normY)
        
        # define additional data for scaling - allows only scaling by lungs
        if self.bodies == 'TorsoLung':
            X_scaling = X[:225]  # num_nodes=337 for LRT, num_nodes=225 for LR
            Y_scaling = Y[:225]  # num_nodes=337 for LRT, num_nodes=225 for LR
            n_scaling, m_scaling = X_scaling.shape
            ny_scaling, my_scaling = Y_scaling.shape
            muX_scaling = X_scaling.mean(0)
            muY_scaling = Y_scaling.mean(0)
            X0_scaling = X_scaling - muX_scaling
            Y0_scaling = Y_scaling - muY_scaling
            X0_full = X - muX_scaling  ##### 
            Y0_full = Y - muY_scaling  #####
            ssX_scaling = (X0_scaling ** 2.).sum()
            ssY_scaling = (Y0_scaling ** 2.).sum()
            normX_scaling = np.sqrt(ssX_scaling)
            normY_scaling = np.sqrt(ssY_scaling)
            X0_scaling /= normX_scaling
            Y0_scaling /= normY_scaling
            X0_full /= normX_scaling
            Y0_full /= normY_scaling            
            if my_scaling < m_scaling:
                Y0_scaling = np.concatenate((Y0_scaling, np.zeros(n_scaling, m_scaling - my_scaling)), 0)
            A_scaling = np.dot(X0_scaling.T, Y0_scaling)
            U_scaling, s_scaling, Vt_scaling = np.linalg.svd(A_scaling, full_matrices=False)
            V_scaling = Vt_scaling.T
            T_scaling = np.dot(V_scaling, U_scaling.T)
            traceTA_scaling = s_scaling.sum()  
            #print("traceTA_scaling: \n", traceTA_scaling)
            #print("normX_scaling: \n", normX_scaling)
            #print("normY_scaling: \n", normY_scaling)

        if scaling:

            if self.bodies == 'TorsoLung':
                """ optimum scaling of Y """
                b = traceTA_scaling * normX_scaling / normY_scaling
                """ standarised distance between X and b*Y*T + c """
                d = 1 - traceTA_scaling ** 2
                """ transformed coords """
                Z = normX_scaling * traceTA_scaling * np.dot(Y0_full, T_scaling) + muX_scaling
                
            else:
                """ optimum scaling of Y """
                b = traceTA * normX / normY
                """ standarised distance between X and b*Y*T + c """
                d = 1 - traceTA ** 2
                """ transformed coords """
                Z = normX * traceTA * np.dot(Y0, T) + muX
                
        else:
            
            if self.bodies == 'TorsoLung':
                b = 1
                d = 1 + ssY_scaling / ssX_scaling - 2 * traceTA_scaling * normY_scaling / normX_scaling
                Z = normY_scaling * np.dot(Y0_full, T_scaling) + muX_scaling
                
            else:
                b = 1
                d = 1 + ssY / ssX - 2 * traceTA * normY / normX
                Z = normY * np.dot(Y0, T) + muX

        """ translation matrix """
        if my < m:
            T = T[:my, :]
        c = muX - b * np.dot(muY, T)

        """ transformation values """
        tform = {'rotation': T, 'scale': b, 'translation': c}

        for num, object in enumerate(self.mesh.nodes):
            node = self.mesh.nodes[object.id].values[:, 0]
            Zlist = Z.tolist()
            self.mesh.nodes[object.id].values[:, 0] = Zlist[num]

        if self.output is not None:
            self.output = None

        self.output = 'morphic_aligned'

        mesh_output = os.path.normpath(mesh + os.sep + os.pardir)
        mesh_output = os.path.normpath(mesh_output + os.sep + os.pardir)

        mesh_output = os.path.join(mesh_output, self.output)

        if not os.path.exists(mesh_output):
            os.makedirs(mesh_output)

        mesh_name = mesh.split("/")[1]

        self.mesh.save(os.path.join(mesh_output, self.bodies+"_fitted.mesh"))

        print('\t   ALIGNED MESH SAVED IN:')
        print('\t   {}\n'.format(mesh_output))

        print('\n\t=========================================\n')

        return d, Z, tform, self.mesh, os.path.join(mesh_output, self.bodies+"_fitted.mesh")


    def align_mesh_ipf(self, reference_mesh, mesh, scaling=True, reflection='best'):
        """
        align_mesh is a method that performes a Procrustes analysis which
        determines a linear transformation (translation, reflection, orthogonal rotation
        and scaling) of the nodes in mesh to best conform them to the nodes in reference_mesh,
        using the sum of squared errors as the goodness of fit criterion.

        Inputs:
        ------------
        reference_mesh, mesh
            meshes (as morphic meshes) of target and input coordinates. they must have equal
            numbers of  nodes (rows), but mesh may have fewer dimensions
            (columns) than reference_mesh.

        scaling
            if False, the scaling component of the transformation is forced
            to 1

        reflection
            if 'best' (default), the transformation solution may or may not
            include a reflection component, depending on which fits the data
            best. setting reflection to True or False forces a solution with
            reflection or no reflection respectively.

        Outputs
        ------------
        d
            the residual sum of squared errors, normalized according to a
            measure of the scale of reference_mesh, ((reference_mesh - reference_mesh.mean(0))**2).sum()

        Z
            the matrix of transformed Y-values

        tform
            a dict specifying the rotation, translation and scaling that
            maps X --> Y

        self.mesh
            Aligned mesh

        :param reference_mesh
        :param mesh
        :param scaling
        :param reflection
        :return: d, Z, tform, self.mesh
        """

        import os

        print('\n\t=========================================\n')
        print('\t   ALIGNING MESH... ')
        #print('\t   PLEASE WAIT... ')

        if self.mesh is not None:
            self.mesh = None

        r = Mesh(reference_mesh)
        self.mesh = Mesh(mesh)

        X = r.get_nodes()
        Y = self.mesh.get_nodes()

        n, m = X.shape
        ny, my = Y.shape

        muX = X.mean(0)
        muY = Y.mean(0)

        X0 = X - muX
        Y0 = Y - muY

        ssX = (X0 ** 2.).sum()
        ssY = (Y0 ** 2.).sum()

        """ centred Frobenius norm """
        normX = np.sqrt(ssX)
        normY = np.sqrt(ssY)

        """ scale to equal (unit) norm """
        X0 /= normX
        Y0 /= normY

        if my < m:
            Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

        """ optimum rotation matrix of Y """
        A = np.dot(X0.T, Y0)
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        V = Vt.T
        T = np.dot(V, U.T)

        if reflection != 'best':

            """ if the current solution use a reflection? """
            have_reflection = np.linalg.det(T) < 0

            """ if that's not what was specified, force another reflection """
            if reflection != have_reflection:
                V[:, -1] *= -1
                s[-1] *= -1
                T = np.dot(V, U.T)

        traceTA = s.sum()

        if scaling:

            """ optimum scaling of Y """
            b = traceTA * normX / normY

            """ standarised distance between X and b*Y*T + c """
            d = 1 - traceTA ** 2

            """ transformed coords """
            Z = normX * traceTA * np.dot(Y0, T) + muX

        else:
            b = 1
            d = 1 + ssY / ssX - 2 * traceTA * normY / normX
            Z = normY * np.dot(Y0, T) + muX

        """ translation matrix """
        if my < m:
            T = T[:my, :]
        c = muX - b * np.dot(muY, T)

        """ transformation values """
        tform = {'rotation': T, 'scale': b, 'translation': c}

        for num, object in enumerate(self.mesh.nodes):
            node = self.mesh.nodes[object.id].values[:, 0]
            Zlist = Z.tolist()
            self.mesh.nodes[object.id].values[:, 0] = Zlist[num]

        if self.output is not None:
            self.output = None

        self.output = 'morphic_aligned'

        mesh_output = os.path.normpath(mesh + os.sep + os.pardir)
        # mesh_output = os.path.normpath(mesh_output + os.sep + os.pardir)

        mesh_output = os.path.join(mesh_output, self.output)

        if not os.path.exists(mesh_output):
            os.makedirs(mesh_output)

        mesh_name = mesh.split("/")[1]

        self.mesh.save(os.path.join(mesh_output, "Lung_fitted.mesh"))

        print('\t   ALIGNED MESH SAVED IN:')
        print('\t   {}\n'.format(mesh_output))

        print('\n\t=========================================\n')

        return d, Z, tform, self.mesh, os.path.join(mesh_output, "Lung_fitted.mesh")

    def convert_cm_mesh(self, file_path, file_path_torso='', bodies='l'):
        """

        :param file_path:
        :param file_name:
        :param bodies:
        :return:
        """
        import os

        if bodies.lower() == 'l':
            output_path, _ = self.process_cm_mesh('Left', file_path)
        elif bodies.lower() == 'r':
            output_path, _ = self.process_cm_mesh('Right', file_path)
        elif bodies.lower() == 'lr':
            output_path, output_lung_left = self.process_cm_mesh('Left', file_path)
            if output_path is None:
                return None
            output_path, output_lung_right = self.process_cm_mesh('Right', file_path)
            if output_path is None:
                return None
            final_file_path = os.path.normpath(output_lung_left + os.sep + os.pardir)
            final_file = os.path.join(final_file_path, 'Lung_fitted.ip2py')

            from time import sleep
            sleep(0.1)
            with open(output_lung_right) as f:
                with open(final_file, "w") as f1:
                    for line in f:
                        f1.write(line)
            with open(output_lung_left) as f:
                with open(final_file, "a") as f1:
                    for line in f:
                        f1.write(line)
                        
        elif bodies.lower() == 'lrt':
            output_path, output_lung_left = self.process_cm_mesh('Left', file_path)
            if output_path is None:
                return None
            output_path, output_lung_right = self.process_cm_mesh('Right', file_path)
            if output_path is None:
                return None
            output_path_torso, output_torso = self.process_cm_mesh('Torso', file_path_torso)         ###########
            if output_path_torso is None:
                return None   
                 
            final_file_path = os.path.normpath(output_lung_left + os.sep + os.pardir)
            final_file = os.path.join(final_file_path, 'TorsoLung_fitted.ip2py')
            
            from time import sleep
            sleep(0.1) # ---------------------- delay here, helps the files keep up with the script when generating meshes!!!
            #print(output_lung_right)
            #print(output_lung_left)
            #print(output_torso)
            #print(final_file)
            
            with open(output_lung_right) as f:
                with open(final_file, "w") as f1:
                    for line in f:
                        f1.write(line)
            with open(output_lung_left) as f:
                with open(final_file, "a") as f2:
                    for line in f:
                        f2.write(line)          
            with open(output_torso) as f:
                with open(final_file, "a") as f3:
                    for line in f:
                        f3.write(line)   
                                   
        return output_path

    def process_cm_mesh(self, body, file_path):
        """

        :param body:
        :return:
        """
        import os
        import subprocess

        #if self.bodies is not None:
        #    self.bodies = None
        #self.bodies = body  ####

        input_folder = 'useful_files'
        output_folder = 'morphic_original'
        output_path = os.path.join(file_path, output_folder)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        perl_file = os.path.join(os.path.dirname(__file__), input_folder, 'perl_com', 'ip2py_%s.pl' % body)
        try:
            _inp = file_path + '/' + body + '_fitted.ipnode'
        except IOError:
            inp = file_path + '/fitted' + body + '.ipnode'
            
        #print("output path:\t", output_path)
        #print("file_path:  \t", file_path)

        if not os.path.exists(_inp):
            print('Mesh for subject {} does not exist. Skipping...'.format(file_path))
            return None, None

        input_body = _inp
        output_body = os.path.join(output_path, body + '_fitted.ip2py')
        #print("input_body: \t", input_body)
        #print("output_body:\t", output_body)
        #print("perl file:    \t", perl_file)       
        subprocess.Popen(["perl", perl_file, input_body, output_body])
        return output_path, output_body
        
        
        