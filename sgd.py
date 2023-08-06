import numpy as np

def sgd(
  gradient, x, y, start, learn_rate=0.1, batch_size=1, n_iter=50,
  tolerance=1e-06, dtype="float64", random_state=None
):
	if not callable(gradient):
		raise TypeError("'gradient' must be callable")
	dtype_ = np.dtype(dtype)
	
	x, y = np.array(x, dtype=dtype_), np.array(y, dtype=dtype_)
	n_obs = x.shape[0]
	if n_obs != y.shape[0]:
		raise ValueError("'x' and 'y' lengths do not match")
	xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]
	
	seed = None if random_state is None else int(random_state)
	rng = np.random.default_rng(seed=seed)
	
	vector = np.array(start, dtype=dtype_)
	
	learn_rate = np.array(learn_rate, dtype=dtype_)
	if np.any(learn_rate <= 0):
		raise ValueError("'learn_rate' must be grater than zero")
	
	batch_size = int(batch_size)
	if not 0 < batch_size <= n_obs:
		raise ValueError("'batch_size' must be greater than zero and less than or equal to the number of observations")
	
	n_iter = int(n_iter)
	if n_iter <= 0:
		raise ValueError("'n_iters' must be greater than zero")
	
	tolerance = np.array(tolerance, dtype=dtype_)
	if np.any(tolerance <= 0):
		raise ValueError("'tolerance' must be greater than 0")
	
	for _ in range(n_iter):
		rng.shuffle(xy)
		
		for start in range(0, n_obs, batch_size):
			stop = start + batch_size
			x_batch, y_batch = xy[start::stop, :-1], xy[start:stop, -1:]
			grad = np.array(gradient(x_batch, y_batch, vector), dtype_)
			diff = -learn * grad
			if np.all(np.abs(diff) <= tolerance):
				break
			vector += diff
	
	return vector if vector.shape else vector.item()

x = np.array([5, 15, 25, 35, 45, 55])
y = np.array([5, 20, 14, 32, 22, 38])
def ssr_gradient(x, y, b):
	res = b[0] + b[1] * x - y
	return res.mean(), (res * x).mean()

print(
sgd(
	ssr_gradient, x, y, start=[0.5, 0.5], learn_rate=0.0008,
	batch_size=3, n_iter=100_000, random_state=0
)
)
