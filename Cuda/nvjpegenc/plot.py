import matplotlib.pyplot as plt
import numpy as np
from glob import glob

fig, ax = plt.subplots()

ax.grid()

data = None
path = "./images/"
contents = glob(f"{path}*.jpeg")

data = [c.replace(path,"").replace(".jpeg","").split("_")[:] for c in contents]
data = sorted(data,key= lambda a : float(a[1]))
print(data)

for i in range(100):
    fdata = list(filter(lambda a : a[0] == str(i),data))
    fdata = np.array(fdata,dtype=np.float)
    if len(fdata.shape) != 2 :
        print(fdata.shape)
        continue
    avg = np.average(fdata[:,1])
    std = np.std(fdata[:,1])
    pixpersecond = 1e6 / np.average(fdata[:,1])
    print(i,pixpersecond, np.abs(pixpersecond - 1e6/(avg + std))  ,np.min(fdata[:,1]),np.max(fdata[:,1]))
    ax.scatter(fdata[:,2]/1000,1e6 / fdata[:,1])

plt.show()

