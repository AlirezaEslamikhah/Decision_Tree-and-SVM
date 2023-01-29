import numpy as np 
import pandas as pd 
from google.colab import files
upload = files.upload()



print("Data Successfully uploaded!\n")
import os
print(os.listdir("../content/"))


import h5py
from functools import reduce
def hdf5(path, data_key="data",target_key="target", flatten=True ):
    with h5py.File(path,'r') as hf:
        train = hf.get('train')
    X_tr = train.get(data_key)[:]
    y_tr = train.get(target_key)[:]
    test = hf.get('test')
    X_te = test.get(data_key)[:]
    y_te = test.get(target_key)[:]
    if flatten:
      X_tr = X_tr.reshape(X_tr.shape[0], reduce(lambda a, b: a * b, X_tr.shape[1:]))
      X_te = X_te.reshape(X_te.shape[0], reduce(lambda a, b: a * b, X_te.shape[1:]))
    return X_tr, y_tr, X_te, y_te

X_tr, y_tr, X_te, y_te = hdf5("../content/usps.h5")
X_tr.shape, X_te.shape


import matplotlib.pyplot as plt
num_samples = 10
num_classes = len(set(y_tr))
# or
classes = set(y_tr)
num_classes = len(classes)

fig, ax = plt.subplots(num_samples, num_classes, sharex = True, sharey = True, figsize=(num_classes, num_samples))

for label in range(num_classes):
    class_idxs = np.where(y_tr == label)
  
    for i, idx in enumerate(np.random.randint(0, class_idxs[0].shape[0], num_samples)):
        ax[i, label].imshow(X_tr[class_idxs[0][idx]].reshape([16, 16]), 'gray')
        ax[i, label].set_axis_off()



from sklearn.svm import LinearSVC
lsvm = LinearSVC(C = 0.1)
lsvm.fit(X_tr, y_tr)

preds = lsvm.predict(X_te)
accuracy = sum(preds == y_te)/len(y_te)
print("Accuracy of Support Vector Machine is", accuracy,"or",round(accuracy*100,2),"%")