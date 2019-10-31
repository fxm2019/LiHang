from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_y)

# print(model.predict(data_X[:4,:]))
# 测试X的前四个数据的预测值
# print(data_y[:4])

# print(model.coef_) #输出feature前面的系数
# print(model.intercept_) #输出截距

# print(model.get_params()) #返回给model定义的参数

print(model.score(data_X, data_y)) #R^2 coeffient of determination

# X,y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=1)
# # n_samples 生成样本数， n_features 样本特征数， noise 样本随机噪声， coef 是否返回回归系数
#
# plt.scatter(X, y)
# # 绘制散点图
# plt.plot(X, y, color='blue', linewidth=2)
#
#
# plt.show()