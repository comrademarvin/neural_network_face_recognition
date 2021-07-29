import numpy as np
import sys
from sklearn.decomposition import PCA

# Initializations
# load data arrays
X_train_upload = np.load(sys.argv[1])
y_train_upload = np.load(sys.argv[2])
X_test_upload = np.load(sys.argv[3])
y_test_upload = np.load(sys.argv[4])

# PCA
X_joined = np.concatenate((X_train_upload,X_test_upload))
X_joined = X_joined/255

pca = PCA(n_components=500)
X_joined_pca = pca.fit_transform(X_joined)

X_splitted = np.split(X_joined_pca, [350])
X_train = X_splitted[0].T
X_test = X_splitted[1].T

# dimension variables
N_train = len(X_train[0])
N_test = len(X_test[0])
D = len(X_train) # input layer
H = 50 # hidden layer
K = 10 # output layer

# construct label matrix as KxN
y_train = np.zeros((K,len(y_train_upload)), int)
y_test = np.zeros((K,len(y_test_upload)),int)

for i in range(len(y_train_upload)):
    y_train[y_train_upload[i],i] = 1
    
for i in range(len(y_test_upload)):
    y_test[y_test_upload[i],i] = 1

# weight matrices
W_1 = np.random.randn(H,D) * 0.01
b_1 = np.random.randn(H,1) * 0.01

W_2 = np.random.randn(K,H) * 0.01
b_2 =  np.random.randn(K,1) * 0.01

# activation functions
def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
	return  np.tanh(x)

def grad_tanh(x):
	return 1 - np.tanh(x)**2

# training loop
for i in range(7500):
	# forward propagation
	Z_1 = np.matmul(W_1,X_train) + b_1
	A_1 = tanh(Z_1)
	Z_2 = np.matmul(W_2,A_1) + b_2
	A_2 = sigmoid(Z_2)

	# cost function
	cost = (1/(N_train*K)) * np.sum(np.sum(-y_train * np.log(A_2) - (1-y_train)* np.log(1 - A_2), axis=0))
	print("Iteration: {}".format(i)," Cost: {:02f}".format(cost))

	# back propogation
	dZ_2 = A_2 - y_train
	dZ_1 = np.matmul(W_2.T,dZ_2) * grad_tanh(Z_1)
	dW_2 = (1/N_train) * np.matmul(dZ_2,A_1.T)
	dW_1 = (1/N_train) * np.matmul(dZ_1,X_train.T)
	db_2 = (1/N_train) * np.sum(dZ_2,axis=1,keepdims=True)
	db_1 = (1/N_train) * np.sum(dZ_1,axis=1,keepdims=True)

	# update weights + biases
	W_1 = W_1 - 0.01*dW_1
	W_2 = W_2 - 0.01*dW_2
	b_1 = b_1 - 0.01*db_1
	b_2 = b_2 - 0.01*db_2
			

accuracy_file = open("Accuracy.txt", "w")
# test set accuracy
# forward propagation
Z_1 = np.matmul(W_1,X_test) + b_1
A_1 = tanh(Z_1)
Z_2 = np.matmul(W_2,A_1) + b_2
A_2 = sigmoid(Z_2)

# confusion matrix
confusion_test = np.zeros((K,K), int)
model = A_2.T

for i in range(N_test):
	# find prediction
	maxIndex = np.where(model[i] == np.amax(model[i]))
	confusion_test[maxIndex[0][0], y_test_upload[i]] = confusion_test[maxIndex[0][0], y_test_upload[i]] + 1

print('Final test set Confusion Matrix:')
print(confusion_test)

# accuracy
top_count = 0
bottom_count = 0
for i in range(K):
	for j in range(K):
		if (i==j):
			top_count = top_count + confusion_test[i,j]
		else:
			bottom_count = bottom_count + confusion_test[i,j]

accuracy = top_count/(top_count+bottom_count)
accuracy_file.write("Test: {:.2f}%\n".format(accuracy*100))
print("Test Accuracy: {:.2f}%".format(accuracy*100))

# train set accuracy
# forward propagation
Z_1 = np.matmul(W_1,X_train) + b_1
A_1 = tanh(Z_1)
Z_2 = np.matmul(W_2,A_1) + b_2
A_2 = sigmoid(Z_2)

# confusion matrix
confusion_train = np.zeros((K,K), int)
model = A_2.T

for i in range(N_train):
	# find prediction
	maxIndex = np.where(model[i] == np.amax(model[i]))
	confusion_train[maxIndex[0][0], y_train_upload[i]] = confusion_train[maxIndex[0][0], y_train_upload[i]] + 1

print('Final Train set Confusion Matrix:')
print(confusion_train)

# accuracy
top_count = 0
bottom_count = 0
for i in range(K):
	for j in range(K):
		if (i==j):
			top_count = top_count + confusion_train[i,j]
		else:
			bottom_count = bottom_count + confusion_train[i,j]

accuracy = top_count/(top_count+bottom_count)
accuracy_file.write("Train: {:.2f}%".format(accuracy*100))
accuracy_file.close()
print("Train Accuracy: {:.2f}%".format(accuracy*100))

# save matrices in text files
np.savetxt('W_1.txt', W_1)
np.savetxt('W_2.txt', W_2)
np.savetxt('b_1.txt', b_1)
np.savetxt('b_2.txt', b_2)

np.savetxt('TestConfusion.txt', confusion_test, fmt='%-2i')
np.savetxt('TrainConfusion.txt', confusion_train, fmt='%-2i')