import os
import numpy as np
import pickle

os.chdir(os.path.dirname(os.path.realpath(__file__)))

#setup data and save
paths_train = ['dataset/dataset/train/no', 'dataset/dataset/train/sphere', 'dataset/dataset/train/vort']
paths_val = ['dataset/dataset/val/no', 'dataset/dataset/val/sphere', 'dataset/dataset/val/vort']
train_imgs = []
test_imgs = []
train_label = []
test_label = []

for path in paths_train:
    for file in os.scandir(path):
        train_imgs.append([np.load(file.path).reshape(150,150,1)])
        if len(train_imgs) <= 10000:
            train_label.append(0)
        elif len(train_imgs) <= 20000:
            train_label.append(1)
        else:
            train_label.append(2)
train_imgs = np.vstack(train_imgs)     

for path in paths_val:
    for file in os.scandir(path):
        test_imgs.append([np.load(file.path).reshape(150,150,1)])
        if len(test_imgs) <= 2500:
            test_label.append(0)
        elif len(test_imgs) <= 5000:
            test_label.append(1)
        else:
            test_label.append(2)
test_imgs = np.vstack(test_imgs)
print(train_imgs.shape)
print(len(train_label))
print(test_imgs.shape)
print(len(test_label))

with open('pickled/train_imgs.pkl', 'wb') as f:
    pickle.dump(train_imgs, f)
with open('pickled/test_imgs.pkl', 'wb') as f:
    pickle.dump(test_imgs, f)
with open('pickled/train_labels.pkl', 'wb') as f:
    pickle.dump(train_label, f)
with open('pickled/test_labels.pkl', 'wb') as f:
    pickle.dump(test_label, f)