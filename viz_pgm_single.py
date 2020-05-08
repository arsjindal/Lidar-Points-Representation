# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:10:21 2020

@author: Vishnu Prem
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

'''
pgm_path = 'E:/pgm_output/08/000208.npy'

pgm = np.load(pgm_path)

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

for i in range(7):
    if i == 6:
        plt.imshow(pgm[...,6:])
        plt.show()
        break
    
    plt.xticks([])
    plt.yticks([])
    plt.imshow(pgm[...,i],)
    plt.show()
'''

num_list = ['000038.npy','000089.npy']

pgm_path = 'E:/model_test_output/normcuts/'+num_list[1]

pgm = np.load(pgm_path, allow_pickle = True)

label_pgm = pgm[...,-1]  
# label_pgm = pgm[0]
img2 = np.zeros((64,512,3), np.uint8)
img2[label_pgm==1] = [0,165,255]
img2[label_pgm==2] = [255,0,255]
img2[label_pgm==3] = [255,255,0]


cv2.imwrite('../img/2d_pred/2d_'+ 'normcuts_2.png', img2)   
cv2.imshow('2d', img2)
        
cv2.waitKey(0)
cv2.destroyAllWindows()

