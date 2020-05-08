
############################
# Loads pgm data from a directory and displays a video of the selected channel 
# Note: Does not work with spyder; run using IDLE
# Author:      Vishnu Prem
#       University of Pennsylvania
#              April 2020
############################

import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import matplotlib.animation as animation

seq_num = 8
# 0:x 1:y 2:z 3:range 4:reflectance 5:label 6-8:rgb 


seq_num_str = str(seq_num) if seq_num > 9 else '0' + str(seq_num)
pgm_dir = 'E:/pgm_output/' + seq_num_str + '/'    
pgm_files = os.listdir(pgm_dir)
num_files = len(pgm_files)

ims = []
fig = plt.figure()
plt.xticks([])
plt.yticks([])

ax1=fig.add_subplot(5,1,1)
ax2=fig.add_subplot(5,1,2)
ax3=fig.add_subplot(5,1,3)
ax4=fig.add_subplot(5,1,4)
ax5=fig.add_subplot(5,1,5)
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')
ax5.axis('off')

for file_num in range(num_files):
    if file_num == 400:
        break
    print(file_num,'/',num_files)
    pgm_path = pgm_dir + pgm_files[file_num]    
    pgm = np.load(pgm_path)

    im1 = ax1.imshow(pgm[...,0], animated=True, cmap =  'gray')
    im2 = ax2.imshow(pgm[...,1], animated=True, cmap =  'gray', vmin = -6, vmax = 6)
    im3 = ax3.imshow(pgm[...,2], animated=True, cmap =  'gray', vmin = -2, vmax = 4)
    im4 = ax4.imshow(pgm[...,3], animated=True, cmap =  'gray')
    im5 = ax5.imshow(pgm[...,4], animated=True, cmap =  'gray')
    
    ims.append([im1, im2, im3, im4, im5])
    
    # if file_num == 100:
    #     break


ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=0)
plt.show()


