
#$pca_id = 'LRT_S_Mfull_N81_R-AGING001-EIsupine';
$pca_id = 'LRT_S_Mfull_N80_R-AGING001-EIsupine_LOO-A';

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
#$version = 'mode05_neg2p5'

#$version = 'sample_mean'
$version = 'predict_H5977'

$nodepath = 'output/export_to_cm/pca_'.$pca_id.'/'.$version;
$elempath = 'useful_files'

gfx read node $nodepath.'/fittedTorso_realigned' reg pca_mesh_Torso;
gfx read elem $elempath.'/templateTorso' reg pca_mesh_Torso;
gfx read node $nodepath.'/fittedLeft_realigned' reg pca_mesh_Left;
gfx read elem $elempath.'/templateLeft' reg pca_mesh_Left;
gfx read node $nodepath.'/fittedRight_realigned' reg pca_mesh_Right;
gfx read elem $elempath.'/templateRight' reg pca_mesh_Right;

gfx cre mat plane_mat ambient 0.7 0.7 0.7 diffuse 0.0 0.0 0.0 specular 0.0 0.0 0.0 alpha 0.4;
gfx mod g_e pca_mesh_Torso surface mat tissue;
gfx mod g_e pca_mesh_Left surface mat muscle;
gfx mod g_e pca_mesh_Right surface mat muscle;
gfx mod g_e pca_mesh_Torso line mat black;
gfx mod g_e pca_mesh_Left line mat black;
gfx mod g_e pca_mesh_Right line mat black;

## CUT PLANE
##gfx read node 'useful_files/cut_plane' reg cut_plane;
##gfx read elem 'useful_files/cut_plane' reg cut_plane;
#gfx mod g_e cut_plane surface mat plane_mat;
##gfx mod g_e cut_plane line none;
#gfx mod g_e cut_plane line mat plane_mat;


gfx edit scene;
gfx cre wind;
gfx node_tool edit select;

#gfx list all_commands
gfx modify window 1 layout simple ortho_axes z -y eye_spacing 0.25 width 768 height 768;
gfx modify window 1 set perturb_lines;
gfx modify window 1 background colour 1 1 1 texture none;
gfx modify window 1 view parallel eye_point -420.697 -409.697 668.681 interest_point 164 175 -158.206 up_vector 0.5 0.5 0.707107 view_angle 40 near_clipping_plane 11.6939 far_clipping_plane 4179.02 relative_viewport ndc_placement -1 1 2 2 viewport_coordinates 0 0 1 1;
gfx modify window 1 overlay scene none;
gfx modify window 1 set transform_tool current_pane 1 std_view_angle 40 perturb_lines no_antialias depth_of_field 0.0 fast_transparency blend_normal;
