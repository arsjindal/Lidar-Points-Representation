# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:15:15 2020

@author: ravit
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm

class PGM:
    
    def __init__(self):
        
        self.proj_H = 64
        self.proj_W = 2048
        
        proj_fov_up = 3
        proj_fov_down = -25.0
        self.fov_up = proj_fov_up / 180.0 * np.pi      # field of view up in rad
        self.fov_down = proj_fov_down / 180.0 * np.pi  # field of view down in rad
        self.fov = abs(self.fov_down) + abs(self.fov_up)  # get field of view total in rad
        self.reset()
           
    def reset(self):
        
        self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
        self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission
          
        self.proj_pgm = np.full((self.proj_H, self.proj_W, 5), -1, dtype=np.float32)     
        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y
        
    def generate_pgm(self, scan):
        
        self.reset()
        # scan = np.fromfile("000000.bin",dtype= np.float32)
        scan = scan.reshape((-1, 4))
        
        self.points = scan[:, 0:3]    # get xyz
        self.remissions = scan[:, 3]  # get remission
               
        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)
        
        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]
        
        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)
        
        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov        # in [0.0, 1.0]
        
        required_range = np.logical_and(yaw>=-np.pi/4, yaw <= np.pi/4)
        
        proj_x = proj_x[required_range]
        proj_y = proj_y[required_range]
        
        self.points = self.points[required_range]
        scan_x = scan_x[required_range]
        scan_y = scan_y[required_range]
        scan_z = scan_z[required_range]
        
        depth = depth[required_range]
        self.remissions = self.remissions[required_range]
            
        # scale to image size using angular resolution
        proj_x *= self.proj_W                              # in [0.0, W]
        proj_y *= self.proj_H                              # in [0.0, H]
        
        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
        
        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
                
        self.proj_pgm[proj_y,proj_x,0:3]  = self.points
        self.proj_pgm[proj_y,proj_x,3]  = self.remissions
        self.proj_pgm[proj_y,proj_x,4]  = depth
        
        self.proj_pgm = self.proj_pgm[:,767:1279,:]
        
        # plt.imshow(proj_pgm[:,:,4])
        # plt.show()

if __name__ == "__main__":
    
    input_dir = '../dataset/sequences/04/velodyne/'
    output_dir = './pgm_output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pgm = PGM()
    
    print('\n Input: ', input_dir)
    print('Output: ', output_dir)
    
    for file_name in tqdm.tqdm(os.listdir(input_dir)):
        
        file_path = input_dir + file_name
        scan = np.fromfile(file_path, dtype= np.float32) 
        pgm.generate_pgm(scan)

        # plt.imshow(pgm.proj_pgm[:,:,4])
        # plt.show()       
        save_path = output_dir + file_name.split('.')[0]
        np.save(save_path, pgm.proj_pgm)
        
        