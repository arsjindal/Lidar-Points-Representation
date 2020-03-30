# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:15:15 2020

@author: ravit
"""

import numpy as np
import matplotlib.pyplot as plt

proj_H = 64
proj_W = 512
proj_fov_up = 3
proj_fov_down = -25.0


points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission
  
proj_pgm = np.full((proj_H, proj_W, 5), -1,
                          dtype=np.float32)
proj_pgm1 = np.full((proj_H, proj_W, 5), -1,
                          dtype=np.float32)

# for each point, where it is in the range image
proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y

scan = np.fromfile("000004.bin",dtype= np.float32)
scan = scan.reshape((-1, 4))


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



required_range = np.logical_and(yaw>=-np.pi/4, yaw <= np.pi/4)
 


proj_x1 = proj_x[required_range]
proj_y1 = proj_y[required_range]


points1 = points[required_range]
scan_x1 = scan_x[required_range]
scan_y1 = scan_y[required_range]
scan_z1 = scan_z[required_range]

depth1 = depth[required_range]
remissions1 = remissions[required_range]

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

# scale to image size using angular resolution
proj_x1 *= proj_W                              # in [0.0, W]
proj_y1 *= proj_H                              # in [0.0, H]

# round and clamp for use as index
proj_x1 = np.floor(proj_x1)
proj_x1 = np.minimum(proj_W - 1, proj_x1)
proj_x1 = np.maximum(0, proj_x1).astype(np.int32)   # in [0,W-1]
proj_x1_copy = np.copy(proj_x1)  # store a copy in orig order

proj_y1 = np.floor(proj_y1)
proj_y1 = np.minimum(proj_H - 1, proj_y1)
proj_y1 = np.maximum(0, proj_y1).astype(np.int32)   # in [0,H-1]
proj_y1_copy = np.copy(proj_y1)  # stope a copy in original order


# assing to images
#polar_grid_map = np.concatenate((points,remissions,depth),axis = 2)
proj_pgm[proj_y,proj_x,0:3]  = points
proj_pgm[proj_y,proj_x,3]  = remissions
proj_pgm[proj_y,proj_x,4]  = depth

proj_pgm1[proj_y1,proj_x1,0:3]  = points1
proj_pgm1[proj_y1,proj_x1,3]  = remissions1
proj_pgm1[proj_y1,proj_x1,4]  = depth1



plt.imshow(proj_pgm[:,:,1])
plt.show()
plt.imshow(proj_pgm1[:,:,1])
plt.show()
