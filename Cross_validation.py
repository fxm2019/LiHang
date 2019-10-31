from sklearn.datasets import load_iris # iris数据集
from sklearn.model_selection import train_test_split # 分割数据模块
from sklearn.neighbors import KNeighborsClassifier # K最近邻(kNN，k-NearestNeighbor)分类算法
from sklearn.model_selection import cross_val_score # K折交叉验证模块
import matplotlib.pyplot as plt

# 交叉验证其实就是将test和train集合多分几次，让他分布的比较平均
# 选择KNN的参数N
# 加载iris数据集
iris = load_iris()
X = iris.data
y = iris.target

k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    loss = -cross_val_score(knn, X, y, cv=10, scoring='neg_mean_squared_error')  # for regression 误差越小越好
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')  # for classification 准确率越高越好
    # cv表示交叉验证的次数
    # k_scores.append(scores.mean())
    k_scores.append(loss.mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

# # 分割数据
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
#
# # 建立模型 n_neighbors=5表示找五个最近邻中最多数的类别当做预测类别
# knn = KNeighborsClassifier(n_neighbors=5)
#
#
# # 训练模型
# knn.fit(X_train, y_train)
#
# # 将准确率打印出
# print(knn.score(X_test, y_test))

#使用K折交叉验证模块
# scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')

#将5次的预测准确率打印出
# print(scores)
# [ 0.96666667  1.          0.93333333  0.96666667  1.        ]

#将5次的预测准确平均率打印出
# print(scores.mean())
# 0.973333333333