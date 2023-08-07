import numpy as np

class SGD:
	def __init__(self, lr=0.01, max_iter=1000, batch_size=32, tol=1e-3):
		# learning rate of the SGD Optimizer
		self.learning_rate = lr
		# maximum number of iterations for SGD Optimizer
		self.max_iteration = max_iter
		# mini-batch size of the data
		self.batch_size = batch_size
		# tolerance for convergence for the theta
		self.tolerence_convergence = tol
		# Initialize model parameters to None
		self.theta = None
		
	def fit(self, X, y):
		# store dimension of input vector
		n, d = X.shape
		# Intialize random Theta for every feature
		self.theta = np.random.randn(d)
		for i in range(self.max_iteration):
			# Shuffle the data
			indices = np.random.permutation(n)
			X = X[indices]
			y = y[indices]
			# Iterate over mini-batches
			for i in range(0, n, self.batch_size):
				X_batch = X[i:i+self.batch_size]
				y_batch = y[i:i+self.batch_size]
				grad = self.gradient(X_batch, y_batch)
				self.theta -= self.learning_rate * grad
			# Check for convergence
			if np.linalg.norm(grad) < self.tolerence_convergence:
				break
	# define a gradient functon for calculating gradient
	# of the data
	def gradient(self, X, y):
		n = len(y)
		# predict target value by taking taking
		# taking dot product of dependent and theta value
		y_pred = np.dot(X, self.theta)
		
		# calculate error between predict and actual value
		error = y_pred - y
		grad = np.dot(X.T, error) / n
		return grad
	
	def predict(self, X):
		# prdict y value using calculated theta value
		y_pred = np.dot(X, self.theta)
		return y_pred
