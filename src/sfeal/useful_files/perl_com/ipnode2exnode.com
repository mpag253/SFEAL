 set echo on;
 fem def param;r;/hpc/mpag253/Torso/SFEAL/src/sfeal/useful_files/perl_com/3d_fitting;
 fem def coor;r;/hpc/mpag253/Torso/SFEAL/src/sfeal/useful_files/perl_com/versions;
 fem def base;r;/hpc/mpag253/Torso/SFEAL/src/sfeal/useful_files/perl_com/BiCubic_Surface_Unit;
 fem def node;r;/hpc/mpag253/Torso/SFEAL/output/export_to_cm/pca_LRT_S_Mfull_N80_R-AGING001-EIsupine_LOO-E/predict_AGING053/fittedTorso;
 fem def elem;r;/hpc/mpag253/Torso/SFEAL/src/sfeal/useful_files/perl_com/templateTorso;
 fem export node;/hpc/mpag253/Torso/SFEAL/output/export_to_cm/pca_LRT_S_Mfull_N80_R-AGING001-EIsupine_LOO-E/predict_AGING053/fittedTorso as fittedTorso;
 fem export elem;/hpc/mpag253/Torso/SFEAL/output/export_to_cm/pca_LRT_S_Mfull_N80_R-AGING001-EIsupine_LOO-E/predict_AGING053/fittedTorso as fittedTorso;
 fem def node;w;/hpc/mpag253/Torso/SFEAL/output/export_to_cm/pca_LRT_S_Mfull_N80_R-AGING001-EIsupine_LOO-E/predict_AGING053/fittedTorso;
 fem quit;
