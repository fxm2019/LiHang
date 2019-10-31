import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import  KNeighborsClassifier

# 使用KNN方法进行分类分析

# 生成数据
centers = [[-2, 2], [2, 2], [0, 4]]
X, y = make_blobs(n_samples=60, centers=centers, random_state=0, cluster_std=0.60)
# 其中cluster_std 是标准差，表示样本点分布的离散程度

# 绘制样本点
plt.figure(figsize=(16, 10))
c = np.array(centers)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='cool')
plt.scatter(c[:, 0], c[:, 1], s=100, marker='^', c='orange')

# 训练模型
k = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X, y)

# 使用模型进行预测
X_sample = [0, 2]
X_sample = np.array(X_sample).reshape(1, -1)
y_sample = clf.predict(X_sample)
neighbors = clf.kneighbors(X_sample, return_distance=False)

#画出示意图
plt.figure(figsize=(16, 10))
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='cool')
plt.scatter(c[:, 0], c[:, 1], s=100, marker='^', c='k')
plt.scatter(X_sample[0][0], X_sample[0][1], marker="x", s=100, cmap='cool')
# 待预测点

for i in neighbors[0]:
    # 绘制预测点和距离最近的五个点之间的连线
    plt.plot([X[i][0], X_sample[0][0]], [X[i][1], X_sample[0][1]], 'k--', linewidth=0.6)

plt.show()
