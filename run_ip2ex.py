import os

pca_id = 'LRT_S_M5_N76_R-AGING025-EIsupine'

#ip_dirs = ['pca_mean_'+pca_id]

#ip_dirs = ['pca_mode5p25_'+pca_id,
#           'pca_mode5n25_'+pca_id,
#           'pca_mode4p25_'+pca_id,
#           'pca_mode4n25_'+pca_id,
#           'pca_mode3p25_'+pca_id,
#           'pca_mode3n25_'+pca_id,
#           'pca_mode2p25_'+pca_id,
#           'pca_mode2n25_'+pca_id,
#           'pca_mode1p25_'+pca_id,
#           'pca_mode1n25_'+pca_id]
           
#ip_dirs = ['pca_predict_H11303-EI_from_'+pca_id]
ip_dirs = ['pca_sample_H11303-EI_from_'+pca_id]

bodies = ['Right', 'Left', 'Torso']

for ip_dir in ip_dirs:
    
    print(ip_dir)

    for body in bodies:
    
        print(body)

        ipread_cmd = 'python3 ipnode2exnode-converter/ipread.py output/export_to_cm/'+ip_dir+'/fitted'+body+'.ipnode temp.json'
        os.system(ipread_cmd)

        exwrite_cmd = 'python3 ipnode2exnode-converter/exwrite.py temp.json output/export_to_cm/'+ip_dir+'/fitted'+body+'.exnode'
        os.system(exwrite_cmd)
    
        #print(ipread_cmd)
        #print(exwrite_cmd)