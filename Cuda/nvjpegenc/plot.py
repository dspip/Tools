import matplotlib.pyplot as plt
import numpy as np
from glob import glob

fig, ax = plt.subplots()

ax.set_xlabel("qp")
ax.set_ylabel("fps")

ax.grid()

data = None
path = "./images/"
contents = None
qp = []
fps = []
skip = 4
with open("timingdata","r") as f:
    contents = f.readlines()
    contents = contents[skip:]
    for c in contents:
        c0 = c.split("_")
        qp.append(float(c0[0]))
        fps.append(1e6/float(c0[1]))

#contents = glob(f"{path}*.jpeg")
ax.plot(qp,fps)

plt.show()

