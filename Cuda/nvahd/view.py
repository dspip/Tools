import cv2
import numpy as np


with open("horz.ppm","rb") as f:
    contents = f.read()
    img = np.reshape(np.frombuffer(contents,np.uint8),(7096,10000,4))[:,:,:]
    img = cv2.cvtColor(img,cv2.COLOR_RGBA2BGRA)

    img = img[:1080,1000:3000,:]

   # cv2.resize(img,(1920,1080))

    cv2.imshow("win",img)
    cv2.waitKey(-1)
