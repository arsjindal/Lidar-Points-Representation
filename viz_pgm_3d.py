
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

# img = cv2.imread('test_img.jpg')
# cv2.imshow('ImageWindow', img)
# cv2.waitKey(0); cv2.destroyAllWindows()
# print(img.shape)

'''
    1. Convert pgm to coods list -done
    2. Compute transformation and transform pts -done
    3. Plot points on image with label
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
    
    CAM_POS = [-5,0,5] # x, y, z
    CAM_INTRINSICS = [0,0,1] #u0, v0, f
    IMG_DIM = [500,1000] # ht, wd
    
    # pgm_path = 'E:/pgm_output/10/000000.npy'
    
    # load pgm
    pgm_dir = 'E:/pgm_output/08/'   
    
    pgm_files = os.listdir(pgm_dir)
    num_files = len(pgm_files)
    
    for file_num in range(num_files):
        print(file_num,'/',num_files)
        pgm_path = pgm_dir + pgm_files[file_num]  
      
        pgm = np.load(pgm_path)
        
        # convert pgm to list of points having (x,y,z,label) and shape: num_points x 4
        points = pgm[...,[0,1,2,5]]
        points_list = points.reshape(-1,4)
        
        # removing points that have no info
        points_list = points_list[points_list[:,0]!=-1]
        
        # homogenous coods and labels
        labels_list = points_list[:,3]
        world_coods_list = points_list[:,:3].T
        world_coods_list = np.vstack((world_coods_list, np.ones((1,world_coods_list.shape[-1]))))
        
        cam_coods = transform_to_cam(world_coods_list, CAM_POS, CAM_INTRINSICS)
        
        # scatter_plot(cam_coods)
                      
        img = np.zeros((IMG_DIM[0],IMG_DIM[1],3), np.uint8)
        image_coods = np.floor(cam_coods*IMG_DIM[0] + IMG_DIM[0]).astype(int)
        image_coods[0] = np.clip(image_coods[0], 0, IMG_DIM[1]-1)
        image_coods[1] = np.clip(image_coods[1], 0, IMG_DIM[0]-1)
        
        car_points = labels_list==1
        person_points = labels_list==2
        bike_points = labels_list==3
        
        img[image_coods[1],image_coods[0],:] = 100
        img[image_coods[1,car_points], image_coods[0,car_points]] = [255,0,255]
        img[image_coods[1,person_points], image_coods[0,person_points]] = [0,255,255]
        img[image_coods[1,bike_points], image_coods[0,bike_points]] = [255,255,0]
        
        cv2.imshow('ImageWindow', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        # print(img.shape)
cv2.destroyAllWindows()