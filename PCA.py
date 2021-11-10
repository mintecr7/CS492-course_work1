import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
import scipy.io
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

input_data = scipy.io.loadmat('face.mat')
X = input_data['X']
L = input_data['l']

# print(L.shape, X.shape)
############ PCA Analysis (Data clean up) ###############
training_size = 8
test_size = 10 - training_size
training_data = np.zeros((2576, training_size * 52), dtype = np.uint8) # 8 columns of training data
test_data = np.zeros((2576, test_size * 52), dtype = np.uint8) # 2 columns of test data
X_shuffle = np.zeros((2576, 520), dtype=np.uint8)
index_train, index_test = 0, 0
zero_centered = True
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

n_dim, n_samples = training_data.shape[0], training_data.shape[1]
X_mean = np.mean(training_data, axis=1, dtype = np.float64)
X_mean =np.array(np.round(X_mean),dtype=np.uint8)
A = training_data - X_mean.reshape((n_dim,1)) if zero_centered else training_data
###### average face #########################
# img = Image.fromarray(X_mean.reshape(46,56), 'L')
# .rotate(-90).show()``
# avgFace = img.rotate(-90)
# plt.title("Avergae face of traning data")
# plt.imshow(avgFace)
###########################################
######### PCA ####################
t0 = time.time()
cov_matrix_high = (1/n_samples) * np.dot(A, A.T)    #covariance matrix high dimensionality
eigenvectors_high, eigenvalues_high, _ = np.linalg.svd(cov_matrix_high, full_matrices=True)
t1 = time.time()

print("High-D execution time: ", t1 - t0)
print("High-D Rank: ", np.linalg.matrix_rank(cov_matrix_high))
print(f"High-D Shape of eigenvectors: {eigenvectors_high.shape}, Shape of eigenvalues: {eigenvalues_high.shape}")

t0 = time.time()
cov_matrix_low = (1/n_samples) * np.dot(A.T, A)     #covariance matrix low dimensionality
eigenvectors_low, eigenvalues_low, _ = np.linalg.svd(cov_matrix_low, full_matrices=True)
print("low shape:" , eigenvectors_low.shape)
t1 = time.time()
print("Low-D execution time: ", t1 - t0)
print("Low-D Rank: ", np.linalg.matrix_rank(cov_matrix_low))
print(f"Low-D Shape of eigenvectors: {eigenvectors_low.shape}, Shape of eigenvalues: {eigenvalues_low.shape}")

non_zero_eigenvalues_high = [x for x in eigenvalues_high if round(x,4) > 0]
non_zero_eigenvalues_low = [x for x in eigenvalues_low if round(x,4) > 0]
# print("High-D: {} Non-zero eigenvalues and {} zero eigenvalues".format(len(non_zero_eigenvalues_high), 2576 - len(non_zero_eigenvalues_high)))
# print("Low-D: {} Non-zero eigenvalues and {} zero eignevalues ".format(len(non_zero_eigenvalues_low), training_size *52 - len(non_zero_eigenvalues_low)))


fig1 = go.Figure()
fig1.add_trace(go.Scatter(y=non_zero_eigenvalues_high,
                    mode='lines',
                    name='High-D'))
fig1.add_trace(go.Scatter(y=non_zero_eigenvalues_low,
                    mode='lines',
                    name='Low-D'))
fig1.update_layout(xaxis_title="Number of eigenvectors with non-zero eigenvalues",
                   yaxis_title="Eigenvalue",
                   width=1100,
                   height=550)
fig1.show()


_ , axes = plt.subplots(3, 10, figsize=(20, 5))
for i, ax in enumerate(axes.flat):
  ax.imshow(eigenvectors_high[:, i].reshape(46, 56).T)

######## Get High-D eigenvectors form Low-D eigenvectors #####
eigenvectors_calculated_from_low = np.dot(A, eigenvectors_low)[:, :-1] #(2576, 415)  U = AV   remove the last vector that gives 0 eigenvalue
eigenvectors_calculated_from_low /= np.array([np.linalg.norm(x) for x in eigenvectors_calculated_from_low.T])   #normalize the vectors
eigenvectors_from_high = eigenvectors_high[:, :len(non_zero_eigenvalues_high)]   #the ones with non-zero eigenvalue

######## Check if eigenvectors obtained from Low-D are identical to eigenvectors of High-D #####

# compare_eigenvectors = np.round(eigenvectors_calculated_from_low - eigenvectors_from_high, decimals=4)
# temp = np.copy(eigenvectors_calculated_from_low)
# arr_1 = np.sum(compare_eigenvectors, axis=0)
# print("Number of nonidentical eigenvectors: ",np.count_nonzero(arr_1))
# for i in range(compare_eigenvectors.shape[1]):
#   if arr_1[i] != float(0):
#     temp[:, i] = np.negative(eigenvectors_calculated_from_low[:, i])    # negate the vector == flip its direction

# compare_eigenvectors_after_flip = np.round(temp - eigenvectors_from_high, decimals=4)
# arr_2 = np.sum(compare_eigenvectors_after_flip, axis=0)
# print("Number of nonidentical eigenvectors: ", np.count_nonzero(arr_2))


####### Working with Top M eigenvectors and Reconstruction #########
M = 300
reconstructed_images = np.zeros(training_data.shape)
top_eigenvectors = eigenvectors_calculated_from_low[:, :M]  #DxM
projection_coeff = np.dot(A.T, top_eigenvectors)  #NxM each row is coefficient of projection of each image in to the space spanned by the top eigenvectors
for i in range(n_samples):
  projected = np.multiply(projection_coeff[i, :], top_eigenvectors)   #a1u1, a2u2, ...
  reconstructed_images[:, i] = X_mean + np.sum(projected, axis=1)   #x_mean + a1u1 + a2u2
### 30 Reconstructed of high-D
_ , axes = plt.subplots(3, 10, figsize=(20, 5))
for i, ax in enumerate(axes.flat):
  ax.imshow(reconstructed_images[:, i].reshape(46, 56).T)
## Sample Reconstruction
fig1 = plt.figure()
img1 = fig1.add_subplot(1, 2, 1)
plt.title("Original")
img1.imshow(training_data[:, 0].reshape(46, 56).T)
img2 = fig1.add_subplot(1, 2, 2)
plt.title("Reconstructed M=" + str(M))
img2.imshow(reconstructed_images[:, 0].reshape(46, 56).T)
if M == 3:
  df = pd.DataFrame(projection_coeff, columns=["x", "y", "z"])
  df["label"] = 0
  j = 1
  for i in range(0, n_samples, training_size):
    df.iloc[i:i+training_size, 3] = j
    j += 1
  sample_images = 5
  fig = px.scatter_3d(df.iloc[:sample_images * training_size, :], x='x', y='y', z='z', color='label', opacity=0.7, title="First " + str(sample_images) + " identites", size_max=0.5, width=1000, height=600)
  fig.update_traces(marker_coloraxis=None)
  fig.show()


projected_train_df = pd.DataFrame(projection_coeff)
projected_train_df["label"] = 0
label = 1
for i in range(0, n_samples, training_size):
  projected_train_df.iloc[i:i+training_size, -1] = label
  label += 1

projected_test_df = pd.DataFrame(np.dot((test_data - X_mean.reshape((n_dim,1))).T, top_eigenvectors)) #104xM
projected_test_df["label"] = 0
label = 1
for i in range(0, test_size * 52, test_size):
  projected_test_df.iloc[i:i+training_size, -1] = label
  label += 1

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(projected_train_df.iloc[:,:-1], projected_train_df["label"])
y_pred = classifier.predict(projected_test_df.iloc[:,:-1])
print("Accuracy of PCA facial detection with M = {}: {}%".format(M, round(accuracy_score(projected_test_df["label"], y_pred)*100, 1)))


fig3 = go.Figure()
for k in [1, 3, 5, 9, 11]:
  vect, acc = [], []
  for M in range(1, 416, 15):
    reconstructed_images = np.zeros(training_data.shape)
    top_eigenvectors = eigenvectors_calculated_from_low[:, :M]  #DxM
    projection_coeff = np.dot(A.T, top_eigenvectors)  #NxM each row is coefficient of projection of each image in to the space spanned by the top eigenvectors
    for i in range(n_samples):
      projected = np.multiply(projection_coeff[i, :], top_eigenvectors)   #a1u1, a2u2, ...
      reconstructed_images[:, i] = X_mean + np.sum(projected, axis=1)   #x_mean + a1u1 + a2u2
    ## Accuracy measurement with M eigenvectors
    projected_train_df = pd.DataFrame(projection_coeff)
    projected_train_df["label"] = 0
    label = 1
    for i in range(0, n_samples, training_size):
      projected_train_df.iloc[i:i+training_size, -1] = label
      label += 1

    projected_test_df = pd.DataFrame(np.dot((test_size - X_mean.reshape((n_dim,1))).T, top_eigenvectors)) #104xM
    projected_test_df["label"] = 0
    label = 1
    for i in range(0, test_size * 52, test_size):
      projected_test_df.iloc[i:i+training_size, -1] = label
      label += 1

    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(projected_train_df.iloc[:,:-1], projected_train_df["label"])
    y_pred = classifier.predict(projected_test_df.iloc[:,:-1])
    vect.append(M)
    acc.append(round(accuracy_score(projected_test_df["label"], y_pred)*100, 1))
  fig3.add_trace(go.Scatter(x=vect, y=acc,
                    mode='lines',
                    name='K='+ str(k)))
# Edit the layout
fig3.update_layout(title='Face detection accuracy for different K and M values',
                   xaxis_title='M',
                   yaxis_title='Accuracy',
                   width=1100,
                   height=550)
fig3.show()


plt.show()