import numpy as np
import cv2
from glob import glob
from math import log10,sqrt

jpegs = glob("./test/*.jpg")

def psnr(source,compressed):
    mse = np.mean((source - compressed)**2)
    maxi = 255
    return 20 * log10(maxi/sqrt(mse))

def read_raw(path,shape):
    with open(path,"rb") as f:
        buff = f.read()
        return np.frombuffer(buff,dtype=np.uint8).reshape(shape)

raw = read_raw("FRAME1",(480,640,3))
qp_to_psnr = {}
for i in jpegs:
    compressed = cv2.imread(i)
    qp = i.split("./test/")[1].split("_")[0]
    psnrv = psnr(raw,compressed)
    qp_to_psnr[qp] = psnrv


print(qp_to_psnr)

qp = list(range(1,100))
for i in range(1,100):

    print(i,qp_to_psnr[str(i)])
