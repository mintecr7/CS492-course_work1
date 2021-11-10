import numpy as np
from numpy.core.fromnumeric import shape
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
training_label1 = []
training_label2 = []
training_label3 = []
training_label4 = []
test_label = []
zero_centered = True
n_samples = training_data.shape[1]
rand_col = np.random.choice(range(10), training_size, replace=False).tolist()
rand_col.sort()
i=0
j=0
while i!=520:
    #   training_data1[:, j] = X[:, i]
    training_label1.append(L[:,i][0])
    i+=1
    #   training_data2[:, j] = X[:, i]
    training_label2.append(L[:,i][0])
    i+=1
    #   training_data3[:, j] = X[:, i]
    training_label3.append(L[:,i][0])
    i+=1
    #   training_data4[:, j] = X[:, i]
    training_label4.append(L[:,i][0])
    i+=1
    test_data[:, j] = X[:, i]
    test_label.append(L[:,i][0])
    i+=1
    j+=1
training_label = training_label1
training_labels = [training_label2, training_label3, training_label4]

for i in range(0, 520, 10):
    pick_train_col = [X[:, i+x] for x in range(training_size)] #rand_col or range(8)
    pick_test_col = [X[:, i+x] for x in range(training_size, 10)] #range(10) if (x not in rand_col) or range(8, 10)
    training_data[:, index_train:index_train + training_size] = np.array(pick_train_col).T
    # test_data[:, index_test:index_test + test_size] = np.array(pick_test_col).T
    X_shuffle[:, index_train + index_test:index_train + index_test + 10] = np.hstack((np.array(pick_train_col).T, np.array(pick_test_col).T))
    index_train += training_size
    index_test += test_size

number_of_partition = 4
partitions = []
for i in range(number_of_partition):
  temp = np.zeros((2576, 104))
  label = []
  k = 0
  for j in range(0, n_samples, training_size):
    temp[:, k:k+2] = training_data[:, j+(i*2) : j+(i*2)+2]
    label.append(L[:, j][0])
    k += 2
  partitions.append(temp)
  training_label.append(label)
print(partitions[0].shape)

def getAvgFace(idx):
    Mean = np.mean(partitions[idx], axis = 1, dtype = np.float64)
    Mean = np.array(np.round(Mean),dtype=np.uint8)
    return partitions[idx] - Mean.reshape(-1,1)


avgFace = [np.mean(partitions[idx], axis=1).reshape(-1,1) for idx in range(4)]
As = [getAvgFace(idx) for idx in range(4)]
A1 = getAvgFace(0)


fig = plt.figure()
img = fig.add_subplot(1,3,2)
plt.title("batch2 avgFace")
img.imshow(np.mean(A1, axis=1).reshape(46,56).T)

Cover1 = np.dot(A1.T,A1) *(1/(4*n_samples))

M = 300

eigenVectors, eigenValue, _ = np.linalg.svd(Cover1, full_matrices=True)

# print(eigenVectors.shape)
####### Get High-D eigenvectors form Low-D eigenvectors #####
eigenvectors_H = np.dot(A1, eigenVectors)[:, :-1] #(2576, 415)  U = AV   remove the last vector that gives 0 eigenvalue
eigenvectors_H /= np.array([np.linalg.norm(x) for x in eigenvectors_H.T])   #normalize the vectors

TopEigenVectors = eigenvectors_H[:, :M]
# print(TopEigenVectors.shape)
### Visualize average face of each partion #######
fig1 = plt.figure()
img1 = fig1.add_subplot(1, 4, 1)
plt.title("batch1 avgFace")
img1.imshow(avgFace[0].reshape(46,56).T)
img2 = fig1.add_subplot(1,4,2)
plt.title("batch2 avgFace")
img2.imshow(avgFace[1].reshape(46,56).T)
img3 = fig1.add_subplot(1,4,3)
plt.title("batch3 avgFace")
img3.imshow(avgFace[2].reshape(46,56).T)
img4 = fig1.add_subplot(1,4,4)
plt.title("batch4 avgFace")
img4.imshow(avgFace[3].reshape(46,56).T)

reconstructed_images = np.zeros(training_data.shape)
projection_coeff = np.dot(A1.T, TopEigenVectors) 
print("projection coefficent shape:",projection_coeff.shape)
Mean1 = np.mean(partitions[0], axis = 1, dtype = np.float64)
Mean1 = np.array(np.round(Mean1),dtype=np.uint8)
for i in range(104):
  projected = np.multiply(projection_coeff[i, :], TopEigenVectors)   #a1u1, a2u2, ...
  reconstructed_images[:, i] = Mean1 + np.sum(projected, axis=1)   #x_mean + a1u1 + a2u2


### 30 Reconstructed of high-D######
_ , axes = plt.subplots(3, 10, figsize=(20, 5))
for i, ax in enumerate(axes.flat):
    plt.title("Reconstructed from the first batch")
    ax.imshow(reconstructed_images[:, i].reshape(46, 56).T)


###### Sample Reconstruction for the first bacth ######
fig1 = plt.figure()
img1 = fig1.add_subplot(1, 2, 1)
plt.title("Original")
img1.imshow(training_data[:, 0].reshape(46, 56).T)
img2 = fig1.add_subplot(1, 2, 2)
plt.title("Reconstructed From the first batch M=" + str(M))
img2.imshow(reconstructed_images[:, 0].reshape(46, 56).T)

for i in range(3):
    next_A = As[i+1]
    next_cover = np.dot(next_A.T, next_A)*(1/(4*n_samples))
    eigenVectors, eigenValue, _ = np.linalg.svd(next_cover, full_matrices=True)

    eigenvectors_H = np.dot(next_A, eigenVectors)[:, :-1] #(2576, 415)  U = AV   remove the last vector that gives 0 eigenvalue
    eigenvectors_H /= np.array([np.linalg.norm(x) for x in eigenvectors_H.T])   #normalize the vectors

    TopEigenVectors = np.concatenate(TopEigenVectors, eigenvectors_H[:, :M])

    training_label += training_labels[i]
    A1 = np.concatenate((A1, next_A))
print(TopEigenVectors.shape)
correct = 0
for test in range(104):
    original_test = test_data[:, test]

    test_face = original_test.reshape(-1,1)-avgFace[0]

    w_test = np.empty(200)

    for i in range(200):
        w_test[i] = (test_face.T)@TopEigenVectors[:, i]

    temp=22222
    picture = 0
    for n in range(208):
        dif = w_test - projected[n, :]
        dif_len = np.linalg.norm(dif)
        if dif_len<temp:
            temp = dif_len
            picture = n

    if training_label[picture]==test_label[test]:
        correct+=1
    



  
accuracy = correct/104

print("Face recognition accuracy of incremental PCA: ", accuracy)

# pri
plt.show()