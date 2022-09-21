from src.sfeal import morphic # mesher
#from src.sfeal.morphic import viewer
#import src.sfeal.morphic.mesher as mmesher
#import src.sfeal.morphic.viewer as mviewer
#import src.sfeal.morphic as morphic

mesh = 'TorsoLung'
subject = 'AGING001'
study = 'Human_Aging'
condition = 'EIsupine'

root = '/hpc/mpag253/Torso/surface_fitting/'
specific = study+'/'+subject+'/'+condition+'/'
fname = mesh+'_nodes_'+subject+'.txt'
mesh = morphic.mesher.Mesh(root+specific+'Lung/SurfaceFEMesh/morphic_aligned/'+mesh+'_fitted.mesh')

Xn = mesh.get_nodes()

#print(Xn)
#with open(fname, 'w') as f:
#    for item in Xn:
#        f.write("%s\n" % item)


Xs, Ts = mesh.get_surfaces(res=32)


S = morphic.viewer.Figure('my_scene', bgcolor=(1,1,1))

S.plot_points('nodes', Xn, color=(1,0,1), size=0.1)
S.plot_surfaces('surface', Xs, Ts, scalars=Xs[:,2])

S.show()

















