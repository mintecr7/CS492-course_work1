import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, PCA
from PIL import Image


input_data = scipy.io.loadmat('face.mat')
X = input_data['X']
L = input_data['l']

training_size = 8
test_size = 10 - training_size
training_data = np.zeros((2576, training_size * 52), dtype = np.uint8) # 8 columns of training data
test_data = np.zeros((2576, test_size * 52), dtype = np.uint8) # 2 columns of test data
X_shuffle = np.zeros((2576, 520), dtype=np.uint8)
index_train, index_test = 0, 0
zero_centered = True
n_samples = training_data.shape[1]
rand_col = np.random.choice(range(10), training_size, replace=False).tolist()
rand_col.sort()
for i in range(0, 520, 10):
    pick_train_col = [X[:, i+x] for x in range(training_size)] #rand_col or range(8)
    pick_test_col = [X[:, i+x] for x in range(training_size, 10)] #range(10) if (x not in rand_col) or range(8, 10)
    training_data[:, index_train:index_train + training_size] = np.array(pick_train_col).T
    test_data[:, index_test:index_test + test_size] = np.array(pick_test_col).T
    X_shuffle[:, index_train + index_test:index_train + index_test + 10] = np.hstack((np.array(pick_train_col).T, np.array(pick_test_col).T))
    index_train += training_size
    index_test += test_size

number_of_partition = 4
partitions = []
for i in range(number_of_partition):
  temp = np.zeros((2576, 104))
  k = 0
  for j in range(0, n_samples, training_size):
    temp[:, k:k+2] = training_data[:, j+(i*2) : j+(i*2)+2]
    k += 2
  partitions.append(temp)
# print(partitions)

def getAvgFace(idx):
    return partitions[idx] - np.mean(partitions[idx], axis=1).reshape(-1,1)


A1 = getAvgFace(0)


fig = plt.figure()
img = fig.add_subplot(1,3,2)
plt.title("batch2 avgFace")
img.imshow(np.mean(A1, axis=1).reshape(46,56).T)

Cover1 = np.dot(A1.T,A1) *(1/(4*n_samples))

M = 50

eigenVectors, eigenValue, _ = np.linalg.svd(Cover1, full_matrices=True)

# print(eigenVectors.shape)
####### Get High-D eigenvectors form Low-D eigenvectors #####
eigenvectors_H = np.dot(A1, eigenVectors)[:, :-1] #(2576, 415)  U = AV   remove the last vector that gives 0 eigenvalue
eigenvectors_H /= np.array([np.linalg.norm(x) for x in eigenvectors_H.T])   #normalize the vectors

TopEigenVectors = eigenvectors_H[:, :M]

### Visualize average face of each partion #######
fig1 = plt.figure()
img1 = fig1.add_subplot(1, 4, 1)
plt.title("batch1 avgFace")
img1.imshow(partitions[0] - np.mean(partitions[0], axis=1).reshape(46,56).T)
img2 = fig1.add_subplot(1,4,2)
plt.title("batch2 avgFace")
img2.imshow(np.mean(partitions[1], axis=1).reshape(46,56).T)
img3 = fig1.add_subplot(1,4,3)
plt.title("batch3 avgFace")
img3.imshow(np.mean(partitions[2], axis=1).reshape(46,56).T)
img4 = fig1.add_subplot(1,4,4)
plt.title("batch4 avgFace")
img4.imshow(np.mean(partitions[3], axis=1).reshape(46,56).T)

# img = ax1.imshow(np.reshape(46,56).T)

# for i in range(3):
#     A_next = getAvgFace(i + 1)
#     next_Cover = np.dot((A_next.T, A_next) *(1/4*(n_samples)))
#     eigenVectors, eigenValue, _ = np.linalg.svd(Cover1, full_matrices=True)

#     eigenvectors_H = np.dot(A1, eigenVectors)[:, :-1] #(2576, 415)  U = AV   remove the last vector that gives 0 eigenvalue
#     eigenvectors_H /= np.array([np.linalg.norm(x) for x in eigenvectors_H.T])   #normalize the vectors

# plt.figure()

plt.show()