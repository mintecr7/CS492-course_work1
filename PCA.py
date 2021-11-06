import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import plotly.express as px
import time
from PIL import Image

input_data = scipy.io.loadmat('face.mat')
X = input_data['X']
L = input_data['l']

# print(L.shape, X.shape)
training_size = 8
test_size = 10 - training_size
training_data = np.zeros((2576, training_size * 52), dtype = np.uint8) # 8 columns of training data
test_data = np.zeros((2576, test_size * 52), dtype = np.uint8) # 2 columns of test data
X_shuffle = np.zeros((2576, 520), dtype=np.uint8)
index_train, index_test = 0, 0
zero_centered = True
rand = np.random.choice(range(10), 8, replace=False).tolist()
rand.sort()


for i in range(0, 520, 10):
    pick_train_col = [X[:, i+x] for x in rand] #rand_col or range(8)
    pick_test_col = [X[:, i+x] for x in range(10) if (x not in rand)] #range(10) if (x not in rand_col) or range(8, 10)
    training_data[:, index_train:index_train + training_size] = np.array(pick_train_col).T
    test_data[:, index_test:index_test + test_size] = np.array(pick_test_col).T
    X_shuffle[:, index_train + index_test:index_train + index_test + 10] = np.hstack((np.array(pick_train_col).T, np.array(pick_test_col).T))
    index_train += training_size
    index_test += test_size



n_dim, n_samples = training_data.shape[0], training_data.shape[1]
X_mean = np.mean(training_data, axis=1, dtype = np.float64)
X_mean =np.array(np.round(X_mean),dtype=np.uint8)

###### average face #########################
img = Image.fromarray(X_mean.reshape(46,56), 'L')
# .rotate(-90).show()
avgFace = img.rotate(-90)
# plt.title("Avergae face of traning data")
# plt.imshow(avgFace)
###########################################

A = training_data - X_mean.reshape((n_dim,1)) if zero_centered else training_data

highD_covariance, lowD_covariance = np.matmul(A, A.T)/416 ,  np.matmul(A.T, A)/416
M = 50

###################################################################
######### high dimenstional PCA ###################################
start_time = time.time()

w1, v1 = np.linalg.eig(highD_covariance)


w1 = -np.sort(-w1)

print("w1:",w1)
# print(c)

end_time = time.time()
totalRun_time = end_time - start_time
###################################################################

###################################################################
######### low dimenstional PCA ###################################
start_time = time.time()

w2, v2 = np.linalg.eig(lowD_covariance)

end_time = time.time()
totalRun_time = end_time - start_time
###################################################################








plt.show()