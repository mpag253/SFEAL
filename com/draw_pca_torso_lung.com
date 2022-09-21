
#$pca_id = 'LRT_S_Mfull_N81_R-AGING001-EIsupine';
$pca_id = 'LRT_S_Mfull_N80_R-AGING001-EIsupine_LOO-E';

#$version = 'mean'
#$version = 'mode01_pos2p5'
#$version = 'mode01_neg2p5'
#$version = 'mode02_pos2p5'
#$version = 'mode02_neg2p5'
#$version = 'mode03_pos2p5'
#$version = 'mode03_neg2p5'
#$version = 'mode04_pos2p5'
#$version = 'mode04_neg2p5'
#$version = 'mode05_pos2p5'
$version = 'mode05_neg2p5'
#$version = 'mode06_pos2p5'
#$version = 'mode06_neg2p5'
#$version = 'mode07_pos2p5'
#$version = 'mode07_neg2p5'
#$version = 'mode08_pos2p5'
#$version = 'mode08_neg2p5'
#$version = 'mode09_pos2p5'
#$version = 'mode09_neg2p5'
#$version = 'mode10_pos2p5'
#$version = 'mode10_neg2p5'

#$version = 'predict_H5977'
#$version = 'predict_AGING043'
#$version = 'predict_H7395'
#$version = 'predict_AGING014'
$version = 'predict_AGING053'

$nodepath = 'output/export_to_cm/pca_'.$pca_id.'/'.$version;
$elempath = 'useful_files'

gfx read node $nodepath.'/fittedTorso' reg pca_mesh_Torso;
gfx read elem $elempath.'/templateTorso' reg pca_mesh_Torso;
gfx read node $nodepath.'/fittedLeft' reg pca_mesh_Left;
gfx read elem $elempath.'/templateLeft' reg pca_mesh_Left;
gfx read node $nodepath.'/fittedRight' reg pca_mesh_Right;
gfx read elem $elempath.'/templateRight' reg pca_mesh_Right;

#gfx cre mat lung_surface ambient 0.4 0.4 0.4 diffuse 0.7 0.7 0.7 specular 0.5 0.5 0.5 alpha 0.4;
gfx mod g_e pca_mesh_Torso surface mat tissue;
#gfx mod g_e pca_mesh_Torso node_points glyph sphere general size "4*4*4" material blue;
gfx mod g_e pca_mesh_Left surface mat muscle;
gfx mod g_e pca_mesh_Right surface mat muscle;

gfx edit scene;
gfx cre wind;
#gfx node_tool edit select;

#open com;edit_derivatives
