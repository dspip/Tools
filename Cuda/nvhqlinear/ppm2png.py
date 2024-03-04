import cv2
import numpy as np

with open("hqlinear_res.ppm","rb") as f:
    contents = f.read()
    img = np.reshape(np.frombuffer(contents,np.uint8),(7096,10000,4))[:,:,:3]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imwrite("hqlinear_res.png",img)
