from calibrate_and_fix import calibrate,test_image
import numpy as np
from exif import Image
import os
from glob import glob

#with open('Images/test3.jpg','rb') as img:
#    i = Image(img)
#    print(i.has_exif,i.get_all())
#mtx = np.load('./calibration_stats/K_4128X3096.npy')
#dist = np.load('./calibration_stats/D_4128X3096.npy')
#print(mtx)
#print(dist)
ret,mtx,dist,rvects,tvecs  = calibrate('SamsungA70',np.array((7,9)),np.array((11,9)), calib_files_dir="./SamsungA70_Stats",remove_failure = True)
#ret,mtx,dist,rvects,tvecs  = calibrate('Note10',np.array((7,9)),np.array((11,9)), calib_files_dir="./Note10_Stats",remove_failure = True)

print(mtx,dist)
#mtx =   1.0e+03 *np.array([
#    [3.7479,0,0],
#    [0,3.7492,0],
#    [2.0900,1.4986,0.0010]])

#k = 1.0e+03
#
#mtx = np.array([[2.7664,0,0],[0,2.7655,0],[2.0021,1.5356,0.0010]],dtype = np.float32)
#distortion = np.array([[0, 0, 0, 0]], dtype=np.float32)
#distortion = np.eye(2)
#
images = glob(os.path.join('/home/dspip/dev/Coffee/coffee-segmentation/Tests1/carpeta fotos/Samsunga70/11440', "*.jpg"))
for i,im in enumerate(images):
    test_image(im,im.replace('.jpg','cal.jpg'),mtx,dist)
#test_image('Images/20211125_104356.jpg','res.jpg',mtx,dist)
