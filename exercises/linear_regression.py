import csv
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 100

def phi(x):
    return [1, x, x**2]

def Phi(X):
    return [phi(xi) for xi in X]

def generate_points():
    x = np.random.uniform(-10, 10, n)
    ep = np.random.normal(0, 2, n)
    return x, [3 * xi + 5 + epsi for xi, epsi in zip(x, ep)]

def linear_regression(x, y):
    w1 = (n * sum(xi * yi for xi, yi in zip(x, y)) - sum(y) * sum(x)) / (n * sum(xi ** 2 for xi in x) - sum(x)**2)
    w0 = (sum(y) - w1 * sum(x)) / n
    return w0, w1

def mean_squared_error(yr, w0, w1, x):
    return 1 / n * sum((yi - (w0 + w1*xi))**2 for yi, xi in zip(yr, x))

def linereg(X, y):
    Xnp = np.array(X)
    ynp = np.array(y)
    return np.linalg.pinv(Xnp.T @ Xnp) @ Xnp.T @ ynp

# x, y = generate_points()

# w0, w1 = linear_regression(x[:80], y[:80])

# xl = np.linspace(-10, 10, 400)
# yl = 3 * xl + 5
# w = w1 * xl + w0
# 
# yr = [3* xi + 5 for xi in x]
# 
# print(mean_squared_error(yr, w0, w1, x[-20:]))
# 
# X = []
# Y = []
# with open('exercises/micro_gas_turbine_eletrical_energy_prediction/train/ex_1.csv', 'r', newline='') as file:
#     reader = csv.DictReader(file)
#     for row in reader:
#         X.append(row['input_voltage'])
#        Y.append(row['el_power'])

#plt.plot(xl, yl, lw=1)
#plt.plot(xl, w, color="red", lw=1)
#plt.scatter(x[:80], y[:80], s=10)
#plt.scatter(x[-20:], y[-20:], s=10, color="red")
#plt.show()

def add_intercept(X):
    Xnp = np.array(X)
    ones = np.ones((Xnp.shape[0], 1))
    return np.hstack([ones, Xnp]) 

def generate_points_quadratic():
    x = np.random.uniform(-10, 10, n)
    ep = np.random.normal(0, 2, n)
    return x, [xi**2 + 9 + epsi for xi, epsi in zip(x, ep)]

X, y = generate_points_quadratic()
# w = linereg(Phi(X), y)

w = linereg(add_intercept(X), y)
 
test = np.random.uniform(-10, 10, n)
testy = np.array(Phi(test)) @ w

plt.scatter(X, y, s=10)
plt.scatter(test, testy, s = 10, color="red")
plt.show()


