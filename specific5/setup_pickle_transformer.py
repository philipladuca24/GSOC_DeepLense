import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.realpath(__file__)))

#setup data and save
paths = ['lenses/no_sub', 'lenses/sub']
imgs = []
labels = []

for path in paths:
    for file in os.scandir(path):
        imgs.append([np.array([plt.imread(file.path)]).reshape(150,150,1)])
        if len(imgs) <= 5000:
            labels.append(0)
        else:
            labels.append(1)
imgs = np.vstack(imgs)     

print(imgs.shape)
print(len(labels))

with open('imgs.pkl', 'wb') as f:
    pickle.dump(imgs, f)
with open('labels.pkl', 'wb') as f:
    pickle.dump(labels, f)