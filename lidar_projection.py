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


def project_lidar2img_plane(scan,pixel_coor,rgb):
	proj_H = 64
	proj_W = 2048
	proj_fov_up = 3
	proj_fov_down = -25.0


	points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
	remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission
	  
	proj_pgm = np.full((proj_H, proj_W, 8), -1,
	                          dtype=np.float32)
	proj_pgm1 = np.full((proj_H, proj_W, 5), -1,
	                          dtype=np.float32)

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
	proj_pgm[proj_y,proj_x,5:]  = rgb[(pixel_coor[1, :]).astype(int),\
										(pixel_coor[0, :]).astype(int)]/255.0
	#proj_pgm = proj_pgm[:,767:1279,:]

	pdb.set_trace()
	plt.imshow(proj_pgm[:,768:1281,4])
	plt.show()

	plt.imshow(proj_pgm[:,768:1281,5:])
	plt.show()

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
	tr_vel[:3] = calib['Tr_velo_to_cam'].reshape(3,4)
	R_0_rect[:3,:3] = calib['R0_rect'].reshape(3,3)
	P2[:3] = calib['P2'].reshape(3,4)
	
	lidar_to_cam = np.matmul(np.matmul(P2,R_0_rect),tr_vel)

	return lidar_to_cam

def find_correspondance(scan,proj_cam2lidar,rgb):

	pts_2d = project_to_image(scan[:,:3].transpose(),proj_cam2lidar)

	img_height, img_width, channels = rgb.shape
	valid_scans = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
	                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
	                    (scan[:,0]>0)
					)[0]
	
	pixel_coor = pts_2d[:, valid_scans]
	
	vel_coor = scan[valid_scans]

	project_lidar2img_plane(vel_coor,pixel_coor,rgb)


def main():
	rgb = cv2.cvtColor(cv2.imread(os.path.join('data/000000.png')), cv2.COLOR_BGR2RGB)

	calib = load_calib('data/calib.txt')

	scan = load_lidar('data/000000.bin')

	proj_cam2lidar = cam_2_lidar(calib)

	find_correspondance(scan, proj_cam2lidar, rgb)

if __name__ == "__main__":
	main()
