from sklearn.model_selection import learning_curve #学习曲线模块
from sklearn.model_selection import validation_curve #学习曲线模块
from sklearn.datasets import load_digits #digits数据集
from sklearn.svm import SVC #Support Vector Classifier
import matplotlib.pyplot as plt #可视化模块
import numpy as np

# detect overfitting
# 利用learning_curve来判断模型处于 overfitting 还是 underfitting
# digits = load_digits()
# # 加载digits数据集，其包含的是手写体的数字，从0到9。
# # 数据集总共有1797个样本，每个样本由64个特征组成，
# # 分别为其手写体对应的8×8像素表示，每个特征取值0~16
#
# X = digits.data
# y = digits.target
#
# train_sizes, train_loss, test_loss = learning_curve(
#     SVC(gamma=0.001), X, y, cv=10, scoring='neg_mean_squared_error',
#     train_sizes=[0.1, 0.25, 0.5, 0.75, 1]
# )
# # 改变SVC的参数gamma可能会使学习曲线变化
#
# train_loss_mean = -np.mean(train_loss, axis=1)
# test_loss_mean = -np.mean(test_loss, axis=1)
#
# plt.plot(train_sizes, train_loss_mean, 'o-', color='r', label='Training')
# plt.plot(train_sizes, test_loss_mean, 'o-', color='g', label='Cross-validation')
#
# plt.xlabel("Training examples")
# plt.ylabel("Loss")
# plt.legend(loc="best")
#
# plt.show()

#digits数据集
digits = load_digits()
X = digits.data
y = digits.target

#建立参数测试集
param_range = np.logspace(-6, -2.3, 5)

#使用validation_curve快速找出参数对模型的影响
train_loss, test_loss = validation_curve(
    SVC(), X, y, param_name='gamma', param_range=param_range, cv=10, scoring='neg_mean_squared_error')

#平均每一轮的平均方差
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

#可视化图形
plt.plot(param_range, train_loss_mean, 'o-', color="r",
         label="Training")
plt.plot(param_range, test_loss_mean, 'o-', color="g",
        label="Cross-validation")

plt.xlabel("gamma")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()