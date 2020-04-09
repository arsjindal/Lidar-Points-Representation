
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
import matplotlib.animation as animation

seq_num = 4

# 0:x 1:y 2:z 3:range 4:reflectance 5:label 6-8:rgb 
channel = 3

seq_num_str = str(seq_num) if seq_num > 9 else '0' + str(seq_num)
pgm_dir = 'E:/pgm_output/' + seq_num_str + '/'    
pgm_files = os.listdir(pgm_dir)
num_files = len(pgm_files)

ims = []
fig = plt.figure()

for file_num in range(num_files):
    print(file_num,'/',num_files)
    pgm_path = pgm_dir + pgm_files[file_num]    
    pgm = np.load(pgm_path)

    im = plt.imshow(pgm[...,4], animated=True)
    ims.append([im])
    
    # if file_num == 100:
    #     break

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
plt.show()
