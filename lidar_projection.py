###############################################################
# 	Lidar Projection code for SqueezeNet
#                   March 2020
#   	Ravi, Vishnu| University of Pennsylvania
#          		
###############################################################

import pdb
import cv2
import os
from math import radians
import numpy as np
import matplotlib.pyplot as plt


def project_lidar2img_plane(scan,pixel_coor,rgb,label,valid_scans):
    
	proj_H = 64
	proj_W = 2048
	proj_fov_up = 3
	proj_fov_down = -25.0


	points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
	remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

    # x,y,z,remission,range,r,g,b,label
	proj_pgm = np.full((proj_H, proj_W, 9), -1,
	                          dtype=np.float32)
	#proj_pgm1 = np.full((proj_H, proj_W, 5), -1,
	#                          dtype=np.float32)

	# for each point, where it is in the range image
	proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
	proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y

	points = scan[:, 0:3]    # get xyz
	remissions = scan[:, 3]  # get remission

	fov_up = proj_fov_up / 180.0 * np.pi      # field of view up in rad
	fov_down = proj_fov_down / 180.0 * np.pi  # field of view down in rad
	fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

	# get depth of all points
	depth = np.linalg.norm(points, 2, axis=1)

	# get scan components
	scan_x = points[:, 0]
	scan_y = points[:, 1]
	scan_z = points[:, 2]

	# get angles of all points
	yaw = -np.arctan2(scan_y, scan_x)
	pitch = np.arcsin(scan_z / depth)
	
	# get projections in image coords
	proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
	proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

	# scale to image size using angular resolution
	proj_x *= proj_W                              # in [0.0, W]
	proj_y *= proj_H                              # in [0.0, H]

	# round and clamp for use as index
	proj_x = np.floor(proj_x)
	proj_x = np.minimum(proj_W - 1, proj_x)
	proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
	proj_x_copy = np.copy(proj_x)  # store a copy in orig order

	proj_y = np.floor(proj_y)
	proj_y = np.minimum(proj_H - 1, proj_y)
	proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
	proj_y_copy = np.copy(proj_y)  # stope a copy in original order


	proj_pgm[proj_y,proj_x,0:3]  = points
	proj_pgm[proj_y,proj_x,3]  = remissions
	proj_pgm[proj_y,proj_x,4]  = depth
	
	proj_pgm[proj_y[valid_scans],proj_x[valid_scans],6:]  = rgb[(pixel_coor[1, :]).astype(int),\
										(pixel_coor[0, :]).astype(int)]/255.0
	proj_pgm[proj_y,proj_x,5]  = label
    
    #pdb.set_trace()
        
# 	plt.imshow(proj_pgm[:,768:1281,8])
# 	plt.show()
# 	plt.imshow(proj_pgm[:,768:1281,5:8])
# 	plt.show()
# 	plt.imshow(proj_pgm[:,768:1281,8])
	return proj_pgm[:,767:1279]
	
    
def load_calib(file_path):
	data = {}
	with open(file_path, 'r') as f:
		for line in f.readlines():
			if len(line)<2:
				continue
			key,values = line.strip().split(':')
			data[key] = np.array([float(value) for value in values.strip().split()])
	return data

def load_lidar(file_path):
	scan = np.fromfile(file_path,dtype=np.float32)
	scan = scan.reshape((-1,4))
	return scan

def project_to_image(points,trans):
	num_pts = points.shape[1]

	points = np.vstack((points, np.ones((1, num_pts))))
	
	points = np.matmul(trans, points)

	points[:2, :] /= points[2, :]
	return points[:2, :]

def cam_2_lidar(calib):
	# Usage: img to lidar
	
	tr_vel, R_0_rect,P2= np.eye(4), np.eye(4), np.eye(4)
	tr_vel[:3] = calib['Tr'].reshape(3,4)
	R_0_rect[:3,:3] = np.eye(3)
	P2[:3] = calib['P2'].reshape(3,4)
	
	lidar_to_cam = np.matmul(np.matmul(P2,R_0_rect),tr_vel)

	return lidar_to_cam

def find_correspondance(scan,proj_cam2lidar,rgb,label):

	pts_2d = project_to_image(scan[:,:3].transpose(),proj_cam2lidar)

	img_height, img_width, channels = rgb.shape
	valid_scans = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
	                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
	                    (scan[:,0]>0)
					)[0]
	
	pixel_coor = pts_2d[:, valid_scans]
	
# 	colors = [[255,255,255],[255,0,0],[0,255,0],[0,0,255]]
# 	valid_labels = label[valid_scans]
# 	label_image = np.zeros(rgb.shape,dtype=np.uint8)
# 	for i in range(pixel_coor.shape[1]):
# 		color = colors[int(valid_labels[i])]
# 		cv2.circle(label_image,(pixel_coor[0,i].astype(int), pixel_coor[1,i].astype(int)),2,color=tuple(color),thickness=-1)
# 	blend = cv2.addWeighted(rgb, 0.7, label_image, 0.3, 1)
# 	plt.imshow(blend)

	vel_coor = scan
	
	return project_lidar2img_plane(vel_coor,pixel_coor,rgb,label,valid_scans)

def load_label(path):
    
    label = np.fromfile(path, dtype= np.uint32)
    label = label.reshape((-1))
    instance_label = label >> 16      # get upper half for instances
    semantic_label = label & 0xFFFF   # get lower half for semantics
    
    new_semantic_label = np.zeros((semantic_label.shape))
    
    # refer: https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
    vehicle_class = [10,13,18,20,252,256,257,258,259]
    twowheeler_class = [11,15,31,32,253,255]
    person_class = [30,254]
    
    replace_labels = {1: vehicle_class,
                      2: twowheeler_class,
                      3: person_class}
    
    for new_label, old_label_group in replace_labels.items():
        for old_label in old_label_group:
            new_semantic_label[semantic_label == old_label] = new_label
    
    return new_semantic_label
 

def get_pgm(scan, rgb, label, calib):
    
    proj_cam2lidar = cam_2_lidar(calib)
    pgm = find_correspondance(scan, proj_cam2lidar, rgb, label)
    return pgm

    
def main():
    
    rgb = cv2.cvtColor(cv2.imread(os.path.join('data/000000.png')), cv2.COLOR_BGR2RGB)
    label = load_label('data/000000.label')
    calib = load_calib('data/calib.txt')
    scan = load_lidar('data/000000.bin')
    
    proj_cam2lidar = cam_2_lidar(calib)
    pgm = find_correspondance(scan, proj_cam2lidar, rgb, label)

    
    fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8) = plt.subplots(8,1)
    
    ax1.set_title('x')
    ax2.set_title('y')
    ax3.set_title('z')
    ax4.set_title('reflectance')
    ax5.set_title('depth')
    ax6.set_title('label')
    ax7.set_title('rgb')
    ax8.set_title('label_projected')
    ax1.imshow(pgm[...,0])
    ax2.imshow(pgm[...,1])
    ax3.imshow(pgm[...,2])
    ax4.imshow(pgm[...,3])
    ax5.imshow(pgm[...,4])
    ax6.imshow(pgm[...,5])
    ax7.imshow(pgm[...,6:])

    color = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)

    blend = cv2.addWeighted(pgm[...,6:]*255, 0.6, color[pgm[...,5].astype(int)]*255, 0.4, 1)
    ax8.imshow(blend.astype(int))
    plt.subplots_adjust(top=1)
    plt.show()

        
if __name__ == "__main__":
 	main()













