# LIDAR Transformations

This repo contains scripts for generating and vizualising data for the project [Semantic Segmentation with LIDAR and RGB](https://github.com/rodri651/RGB_Lidar_Segmentation). The project uses data [Semantic KITTI Dataset](http://semantic-kitti.org/) to train Deep Neural Networks on the Segmentation task.

## Generating the data

The Semantic KITTI Data is in a format that is not suitable for inputing to a Neural Network. We convert the raw data into a Polar Grid Map(PGM) which is essentailly a spherical projection of the 360 degree LIDAR. Since our project focuses on combining the RGB and LIDAR data to make predictions, one LIDAR points within the field of view of the camera are considered. The PGM has multiple channels which include x, y, z, intensity, depth, R, G, B and ground truth labels. The image below shows a RGB image sample from the data and the correspdonging PGM that was generated.

<p align="center"> 
<img src="/img/pgm_channels.png" width = "200">
</p>

To generate the PGM run
```
python3 gen_pgm_data.py
```
Ensure that the following point to the right directories
```
RGB_DIR = 'E:/data_odometry_color/dataset/sequences/' 
LABEL_DIR = 'E:/data_odometry_labels/sequences/' 
SCAN_DIR = 'E:/data_odometry_velodyne/dataset/sequences/' 
CALIB_DIR = 'E:/data_odometry_calib/dataset/sequences/'
```
and set the following to the sequence number for which you want to generate the PGM
```
SEQ_NUM = 4
```
## Vizualising the PGM

### 2D
To vizualise the generated PGM data in 2d run
```
python3 viz_pgm_video.py
```
Ensure that the path is set correctly
```
pgm_dir: directory containing all the generated PGM .npy files
```
The result should look like this

<p align="center"> 
<img src="/img/pgm_viz2.gif" width = "400"/>
</p> 

### 3D

To vizualize the PGM in 3D run
```
python3 viz_pgm_3d.py
```
Ensure that the required directories have been selected
```
xyz_pgm_dir: path to directory containing PGM data
label_pgm_dir: path to directory containing model prediction/PGM data(for ground truth)
rgb_dir: path to directory containing corresponding RGB images
```
The result should look something like this


<p align="center"> 
<img src="/img/2d_ground_truth.gif" width = "400"/>
</p> 

<p align="center"> 
<img src="/img/3d_ground_truth.gif" width = "400"/>
</p> 
