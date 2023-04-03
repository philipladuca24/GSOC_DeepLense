import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.realpath(__file__)))

#setup data and save
paths = ['lens_data_alt/lens_data']
imgs = []
labels = []

for path in paths:
    for file in os.scandir(path):
        temp = np.load(file.path, allow_pickle=True)
        imgs.append([np.array([temp[0]]).reshape(150,150,1)])
        labels.append(temp[1])
imgs = np.vstack(imgs)

print(imgs.shape)
print(len(labels))

with open('mass_imgs.pkl', 'wb') as f:
    pickle.dump(imgs, f)
with open('mass_labels.pkl', 'wb') as f:
    pickle.dump(labels, f)