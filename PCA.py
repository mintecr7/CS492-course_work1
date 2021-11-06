import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
# .rotate(-90).show()``
avgFace = img.rotate(-90)
# plt.title("Avergae face of traning data")
# plt.imshow(avgFace)
###########################################

A = training_data - X_mean.reshape((n_dim,1)) if zero_centered else training_data

def Our_PCA(low_D=True):
  cov_matrix = (1/n_samples) * np.matmul(A.T, A) if low_D else (1/n_samples) * np.matmul(A, A.T)    #choose dimensionality
  u, s = np.linalg.eig(cov_matrix)
  return u, s 

highD_covariance, lowD_covariance = np.matmul(A, A.T)/416 ,  np.matmul(A.T, A)/416
M = 50

###################################################################
######### high dimensional PCA ###################################
start_time = time.time()

eig_val1, eig_vec1 = Our_PCA(False)

index_eig1 = {}

for i in range(len(eig_val1)):
    index_eig1[eig_val1[i]] = i

eig_val1 = -np.sort(-eig_val1)

top_M_eigVal1 = eig_val1[:M]
top_M_eigVec1 = np.zeros([2576, M])

for i in range(M):
    top_M_eigVec1[:, i] = eig_vec1[:,index_eig1[top_M_eigVal1[i]]]

fig1 = go.Figure()
fig1.add_trace(go.Scatter(y=top_M_eigVal1[:20],
                    mode='lines',
                    name='High-D'))

fig2 = plt.figure()
ax1 = fig2.add_subplot(1, 3, 1)
img = ax1.imshow(np.reshape(top_M_eigVec1[:, 3],(46,56)).T)



img2 = fig2.add_subplot(1, 3, 2)
img2.imshow(np.reshape(top_M_eigVec1[:, 4],(46,56)).T)



img3 = fig2.add_subplot(1, 3, 3)
img3.imshow(np.reshape(top_M_eigVec1[:, 5],(46,56)).T)


end_time = time.time()
totalRun_time = end_time - start_time
print("High Dimebsional PCA eigenvector shape:", top_M_eigVec1.shape)
print("High Dimensional PCA time taken:", totalRun_time)
###################################################################



###################################################################
######### low dimensional PCA ###################################
start_time = time.time()

eig_val2, eig_vec2 = Our_PCA(True)

index_eig2 = {}

for i in range(len(eig_val2)):
    index_eig2[eig_val2[i]] = i

eig_val2 = -np.sort(-eig_val2)

top_M_eigVal2 = eig_val2[:M]
top_M_eigVec2 = np.zeros([416, M])

for i in range(M):
    top_M_eigVec2[:,i] = eig_vec2[:,index_eig2[top_M_eigVal2[i]]]

fig1.add_trace(go.Scatter(y=top_M_eigVal2[:20],
                    mode='lines',
                    name='Low-D'))

end_time = time.time()
totalRun_time = end_time - start_time
print("low Dimebsional PCA eigenvector shape:", top_M_eigVec2.shape)
print("low Dimensional PCA time taken:", totalRun_time)
###################################################################

fig1.update_layout(xaxis_title="Number of eigenvectors with non-zero eigenvalues",
                   yaxis_title="Eigenvalue")
fig1.show()



plt.show()