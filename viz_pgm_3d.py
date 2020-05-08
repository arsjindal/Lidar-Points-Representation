
############################
# Loads pgm data from a directory and classified 3Dlidar points on an image plane 
# Author:      Vishnu Prem
#       University of Pennsylvania
#              May 2020
############################

import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import os
from numpy.linalg import norm 
import time

'''
    1. Convert pgm to coods list -done
    2. Compute transformation and transform pts -done
    3. Plot points on image with label- done
'''
    
def get_transformation_matrix():
    T = np.array([[ 0., -1.,  0.,  0.],
                  [-1.,  0.,  0.,  0.],
                  [ 0.,  0., -1., 10.]])
                  # [ 0.,  0.,  0.,  1.]])
    return T

def get_tranformation_from_pt(p):
    p = np.array(p)
    tz = norm(p)
    z_cam = -p/norm(p)
    
    x2_cam_1 = z_cam[0]/norm(z_cam[:2])
    x2_cam_2 = -z_cam[0]/norm(z_cam[:2])
    
    x1_cam_1 = -x2_cam_1*z_cam[1]/z_cam[0]
    x1_cam_2 = -x2_cam_2*z_cam[1]/z_cam[0]
    
    x_cam_1 = np.array([x1_cam_1, x2_cam_1, 0])
    x_cam_2 = np.array([x1_cam_2, x2_cam_2, 0])
    
    y_cam_1 = np.cross(z_cam, x_cam_1)
    y_cam_2 = np.cross(z_cam, x_cam_2)
    
    if y_cam_1[2] < 0:
        x_cam = x_cam_1
        y_cam = y_cam_1
    else:
        x_cam = x_cam_2
        y_cam = y_cam_2
        
    # print(x_cam, y_cam, z_cam)
    axes_cam = [x_cam, y_cam, z_cam]
    axes_lidar = [[1,0,0],[0,1,0],[0,0,1]]
    
    rot_matrix = np.zeros((3,3))
    for row in range(3):
        for col in range(3):
            rot_matrix[row,col] = np.dot(axes_cam[row], axes_lidar[col])
            
    T_matrix = np.zeros((3,4))
    T_matrix[:3,:3] = rot_matrix
    T_matrix[2,-1] = tz
    # T_matrix[-1,-1] = 1
    
    # test_pt = np.array([[0.],[0.],[0.],[1.]])
    # print(T_matrix@test_pt)
    
    return T_matrix
    
    
def transform_to_cam(world_coods, cam_pos, cam_intrinsics):   
    
    T = get_tranformation_from_pt(cam_pos)
    
    u0, v0, f = cam_intrinsics
    K = np.array([[f, 0., u0],
                  [0., f, v0],
                  [0., 0., 1.]])
    
    cam_coods =  K @ T @ world_coods
    cam_coods = cam_coods/cam_coods[2]
    
    return cam_coods[:2]
    # return image_coods

def scatter_plot(cam_coods):
        plt.scatter(cam_coods[0],-cam_coods[1],0.001)
        # plt.rcParams['figure.facecolor'] = 'black'
        plt.axis('scaled')
        plt.ylim(0,1)
        plt.xlim(-1,1)
        plt.show()


if __name__ == '__main__':    
    
    
    CAM_POS = [-5,0,5] # x, y, z [5,3,7]
    CAM_INTRINSICS = [500,500,500] #u0, v0, f # [900,500,400] 
    IMG_DIM = [500,1000] # ht, wdq
    
    # pgm_path = 'E:/pgm_output/10/000000.npy'
    
    # load pgm
    xyz_pgm_dir = 'E:/pgm_output/08/'   
    # label_pgm_dir = 'E:/pgm_output/08/'
    label_pgm_dir = 'E:/model_test_output/normcut/renamed/' 
    rgb_dir = 'E:/data_odometry_color/dataset/sequences/08/image_2/'
    video_name = '../videos/xyz'
    
    xyz_pgm_files = os.listdir(xyz_pgm_dir)
    rgb_files = os.listdir(rgb_dir)
    label_pgm_files = os.listdir(label_pgm_dir)
    
    num_files = len(label_pgm_files)
    video=cv2.VideoWriter(video_name+'.mp4',-1,8,(IMG_DIM[1],IMG_DIM[0]))
    
    video2=cv2.VideoWriter('../videos/2d_effnet.mp4',-1,8,(512,64))
    
    
    for file_num in range(num_files):
        
        if file_num == 401:
            break
        
        print(file_num,'/',num_files, " ",label_pgm_files[file_num])
        xyz_pgm_path = xyz_pgm_dir + xyz_pgm_files[file_num]  
        rgb_path = rgb_dir + rgb_files[file_num]
        label_pgm_path = label_pgm_dir + label_pgm_files[file_num]
        
        xyz_pgm = np.load(xyz_pgm_path)
        label_pgm = np.load(label_pgm_path,allow_pickle = True)
        rgb = cv2.imread(rgb_path)
        
        # convert pgm to list of points having (x,y,z,label) and shape: num_points x 4
        
        xyz_pgm = xyz_pgm[...,:3]
        label_pgm = np.expand_dims(label_pgm[...,-1], 2) if label_pgm.shape[-1]==4 else np.expand_dims(label_pgm[...,5], 2)
        
        # label_pgm = np.expand_dims(label_pgm[0],2)
        
        points = np.concatenate((xyz_pgm, label_pgm),2)
        
        # points = pgm[...,[0,1,2,5]] if pgm.shape[-1]==9 else pgm[...,[0,1,2,-1]]
        
        points_list = points.reshape(-1,4)
        
        # removing points that have no info
        points_list = points_list[points_list[:,0]!=-1]
        
        # homogenous coods and labels
        labels_list = points_list[:,3]
        world_coods_list = points_list[:,:3].T
        world_coods_list = np.vstack((world_coods_list, np.ones((1,world_coods_list.shape[-1]))))
        
        image_coods = transform_to_cam(world_coods_list, CAM_POS, CAM_INTRINSICS)
        
        # scatter_plot(cam_coods)
        # labelling
        img = np.zeros((IMG_DIM[0],IMG_DIM[1],3), np.uint8) 
        image_coods = np.floor(image_coods).astype(int)
        image_coods[0] = np.clip(image_coods[0], 0, IMG_DIM[1]-1)
        image_coods[1] = np.clip(image_coods[1], 0, IMG_DIM[0]-1)
        
        car_points = labels_list==1
        person_points = labels_list==3
        bike_points = labels_list==2
        
        img[image_coods[1],image_coods[0],:]                            = 255
        img[image_coods[1,car_points], image_coods[0,car_points]]       = [0,165,255]
        img[image_coods[1,person_points], image_coods[0,person_points]] = [255,0,255]
        img[image_coods[1,bike_points], image_coods[0,bike_points]]     = [255,255,0]
        
        #overlay rgb image
        # dim = (1200,400)
        
        # resize_rgb = cv2.resize(rgb, dim, interpolation = cv2.INTER_AREA)
        # cv2.imshow('rgbWindow', rgb)
        # img[-100:,0:300] = resize_rgb
               
        video.write(img)
        
        img2 = np.zeros((64,512,3), np.uint8)
        img2[label_pgm[...,0]==1] = [0,165,255]
        img2[label_pgm[...,0]==2] = [255,0,255]
        img2[label_pgm[...,0]==3] = [255,255,0]
        
        video2.write(img2)
        
        cv2.imshow('3d', img)
        cv2.imshow('2d', img2)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
        
        # print(img.shape)
cv2.destroyAllWindows()
video2.release()
video.release()