import numpy as np
import matplotlib.pyplot as plt

n_dots = 20

x = np.linspace(0, 1, n_dots)
y = np.sqrt(x) + 0.2*np.random.rand(n_dots) - 0.1

def plot_polynomial_fit(x, y, order):
    p = np.poly1d(np.polyfit(x, y, order))
    # 对函数进行多项式拟合,poly1d的参数是多项式函数的系数

    t = np.linspace(0, 1, 200)
    plt.plot(x, y, 'ro', t, p(t), 'g-', t, np.sqrt(t), 'r--')
    return p

plt.figure(figsize=(18, 4))
titles = ['Under fitting', 'Fitting', 'Over fitting']
models = [None, None, None]
for index, order in enumerate([1, 3, 10]):
    plt.subplot(1, 3, index + 1)
    models[index] = plot_polynomial_fit(x, y, order)
    plt.title(titles[index], fontsize=20)


for m in models:
    print('model coeffs:{0}'.format(m.coeffs))

coeffs_1d = [0.2, 0.6]

plt.figure(figsize=(9, 6))
t = np.linspace(0, 1, 200)
plt.plot(x, y, 'ro', t, models[0](t), '-', t, np.poly1d(coeffs_1d)(t), 'r-')
plt.annotate(r'L1: $y = {1} + {0}x$'.format(coeffs_1d[0], coeffs_1d[1]),
             xy=(0.8, np.poly1d(coeffs_1d)(0.8)), xycoords='data',
             xytext=(-90, -50), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.annotate(r'L2: $y = {1} + {0}x$'.format(models[0].coeffs[0], models[0].coeffs[1]),
             xy=(0.3, models[0](0.3)), xycoords='data',
             xytext=(-90, -50), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.show()