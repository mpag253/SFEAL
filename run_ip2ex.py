import os

#pca_id = 'LRT_S_Mfull_N81_R-AGING001-EIsupine'
pca_id = 'LRT_S_Mfull_N80_R-AGING001-EIsupine'

#ip_dirs = ['pca_'+pca_id+'/mean']

#ip_dirs = ['pca_'+pca_id+'/mean',
#           'pca_'+pca_id+'/mode01_pos2p5',
#           'pca_'+pca_id+'/mode01_neg2p5',
#           'pca_'+pca_id+'/mode02_pos2p5',
#           'pca_'+pca_id+'/mode02_neg2p5',
#           'pca_'+pca_id+'/mode03_pos2p5',
#           'pca_'+pca_id+'/mode03_neg2p5',
#           'pca_'+pca_id+'/mode04_pos2p5',
#           'pca_'+pca_id+'/mode04_neg2p5',
#           'pca_'+pca_id+'/mode05_pos2p5',
#           'pca_'+pca_id+'/mode05_neg2p5',
#           'pca_'+pca_id+'/mode06_pos2p5',
#           'pca_'+pca_id+'/mode06_neg2p5',
#           'pca_'+pca_id+'/mode07_pos2p5',
#           'pca_'+pca_id+'/mode07_neg2p5',
#           'pca_'+pca_id+'/mode08_pos2p5',
#           'pca_'+pca_id+'/mode08_neg2p5',
#           'pca_'+pca_id+'/mode09_pos2p5',
#           'pca_'+pca_id+'/mode09_neg2p5',
#           'pca_'+pca_id+'/mode10_pos2p5',
#           'pca_'+pca_id+'/mode10_neg2p5',]
           
#ip_dirs = ['pca_'+pca_id+'_LOO-A/predict_H5977',
#           'pca_'+pca_id+'_LOO-B/predict_AGING043',
#           'pca_'+pca_id+'_LOO-C/predict_H7395',
#           'pca_'+pca_id+'_LOO-D/predict_AGING014',
#           'pca_'+pca_id+'_LOO-E/predict_AGING053',]
           
ip_dirs = ['pca_'+pca_id+'_LOO-A/sample_mean',
           'pca_'+pca_id+'_LOO-B/sample_mean',
           'pca_'+pca_id+'_LOO-C/sample_mean',
           'pca_'+pca_id+'_LOO-D/sample_mean',
           'pca_'+pca_id+'_LOO-E/sample_mean',]

bodies = ['Right', 'Left', 'Torso']

for ip_dir in ip_dirs:
    
    print(ip_dir)

    for body in bodies:
    
        print(body)

        ipread_cmd = 'python3 ipnode2exnode-converter/ipread.py output/export_to_cm/'+ip_dir+'/fitted'+body+'.ipnode temp/temp.json'
        os.system(ipread_cmd)

        exwrite_cmd = 'python3 ipnode2exnode-converter/exwrite.py temp/temp.json output/export_to_cm/'+ip_dir+'/fitted'+body+'.exnode'
        os.system(exwrite_cmd)
    
        #print(ipread_cmd)
        #print(exwrite_cmd)