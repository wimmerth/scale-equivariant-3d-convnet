import numpy as np

# Mean channel means and variances of MICCAI dataset when splitting dataset after 250 samples
mean_train = np.array([68.87056104, 74.17433623, 75.4319097, 40.57870797], dtype=float)
var_train = np.array([45493.38768838, 55603.72357109, 55807.27523082, 13371.67752623], dtype=float)
mean_train_nonzero = np.array([424.65736352, 457.95341726, 463.27740861, 251.13047884], dtype=float)
var_train_nonzero = np.array([21407.55755011, 30940.76193525, 67064.4517549, 10421.92450364], dtype=float)
mean_test_nonzero = np.array([1181.56245408, 1460.76501836, 1405.64131696, 896.26746974], dtype=float)
var_test_nonzero = np.array([410103.98168446, 408411.21624156, 792125.96880652, 438701.65102767], dtype=float)
mean_test = np.array([195.27322325, 243.98717042, 233.05995117, 148.94592902], dtype=float)
var_test = np.array([689787.30711645, 837808.2972774, 1016120.48723981, 774051.46423515], dtype=float)
