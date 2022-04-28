
$pca_id = 'LRT_S_M5_N76_R-AGING025-EIsupine';

#$version = 'pca_mean_'
#$version = 'pca_mode1p25_'
#$version = 'pca_mode1n25_'
#$version = 'pca_mode2p25_'
#$version = 'pca_mode2n25_'
#$version = 'pca_mode3p25_'
#$version = 'pca_mode3n25_'
#$version = 'pca_mode4p25_'
#$version = 'pca_mode4n25_'
#$version = 'pca_mode5p25_'
#$version = 'pca_mode5n25_'
#$version = 'pca_predict_H11303-EI_from_'
$version = 'pca_sample_H11303-EI_from_'

$nodepath = 'output/export_to_cm/'.$version.$pca_id;
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
