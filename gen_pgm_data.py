###############################################################
# 	Data Generation for SqueezeNet
#                   March 2020
#   	Vishnu| University of Pennsylvania	
###############################################################


import lidar_projection as lidar_proj
import os
import tqdm
import numpy as np
import cv2

       
if __name__ == "__main__": 
    
    seq_num = 4
    seq_num_str = str(seq_num) if seq_num > 9 else '0'+str(seq_num)
    gen_single = False
    file_to_gen = 0
    
    rgb_dir = 'E:/data_odometry_color/dataset/sequences/' + seq_num_str +'/image_2/'
    label_dir = 'E:/data_odometry_labels/sequences/' + seq_num_str +'/labels/'
    scan_dir = 'E:/data_odometry_velodyne/dataset/sequences/' + seq_num_str +'/velodyne/'   
    calib_path = 'E:/data_odometry_calib/dataset/sequences/' + seq_num_str +'/calib.txt'
    
    output_dir = 'E:/pgm_output/' + seq_num_str + '/'
    
    print('Generating PGM for sequence ', seq_num_str)
    
    print('\nLoading rgb from ', rgb_dir)
    print('Loading labels from ', label_dir)
    print('Loading lidar scans from ', scan_dir)
    print('Loading caliberation from ', calib_path)
    print('\nSaving PGM data to ', output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    rgb_files = os.listdir(rgb_dir)
    label_files = os.listdir(label_dir)
    scan_files = os.listdir(scan_dir)
    
    if not(len(rgb_files) == len(label_files) == len(scan_files)):
        print('rgb,label,scan: ', len(rgb_files), len(label_files), len(scan_files))
        raise ValueError('Number of files in source directories not equal!')
        
    num_files = len(rgb_files)
    calib = lidar_proj.load_calib(calib_path)
    
    print('Total timesteps: ', num_files)
    
    for file_num in tqdm.tqdm((range(num_files))):
        
        if gen_single and file_num != file_to_gen:
            continue
        
        rgb_path = rgb_dir + rgb_files[file_num]
        label_path = label_dir + label_files[file_num]
        scan_path = scan_dir + scan_files[file_num]
        
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        label = lidar_proj.load_label(label_path)
        scan = lidar_proj.load_lidar(scan_path)
        
        pgm = lidar_proj.get_pgm(scan, rgb, label, calib)
        
        output_path = output_dir + rgb_files[file_num].split('.')[0]
        np.save(output_path, pgm)
        
        if gen_single: 
            break    
        
    print('\nPGM data at ', output_dir)
    
    
    
    
    
    
    