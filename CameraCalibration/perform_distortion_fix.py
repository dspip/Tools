
import numpy as np
from calibrate_and_fix import test_image

#these parameters should be updated accordingly to the output from matlab
mtx = np.array([[2.7664,0,0],[0,2.7655,0],[2.0021,1.5356,0.0010]],dtype = np.float32)
dist = np.array([[0, 0, 0, 0]], dtype=np.float32)

test_image('res.jpg','2.jpg',mtx,dist)
