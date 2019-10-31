import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.feature_selection import SelectKBest


# 加载数据
data = pd.read_csv('diabetes.csv')
# 数据的前八列是features，最后一列是label

# print("dataset shape:{}".format(data.shape))
# print(data.head(10))

# print(data.groupby("Outcome").size())

X = data.iloc[:, 0:8]
y = data.iloc[:, 8]
print("shape of X:{},shape of y:{}".format(X.shape, y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 建立三个模型
models = []
models.append(("KNN", KNeighborsClassifier(n_neighbors=2)))
models.append(("KNN with weights", KNeighborsClassifier(n_neighbors=2, weights="distance")))
models.append(("Radius Neighbors", RadiusNeighborsClassifier(n_neighbors=2, radius=500.0)))

# 分别训练三个模型
# 交叉验证
results = [] # 保存每个模型的得分
# for name, model in models:
#     # kfold = KFold(n_splits=10)
#     # cv_result = cross_val_score(model, X, y, cv=10)
#     model.fit(X_train, y_train)
#     results.append((name, model.score(X_test, y_test)))
#
# for i in range(len(results)):
#     print("name:{}, score:{}".format(results[i][0], results[i][1].mean()))

# 加上交叉验证
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_result = cross_val_score(model, X, y, cv=10)
    model.fit(X_train, y_train)
    results.append((name, cv_result))

for i in range(len(results)):
    print("name:{}, cross val score:{}".format(results[i][0], results[i][1].mean()))

# 经过上面的交叉对比，发现最简单的KNN模型得到的准确率最高
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
train_score = knn.score(X_train, y_train)
test_score = knn.score(X_test, y_test)
print("train score: {}; test score: {}".format(train_score, test_score))


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o--', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plt.figure(figsize=(10, 6))
plot_learning_curve(knn, "Learn Curve for KNN Diabetes", X, y, ylim=(0.0, 1.01), cv=cv)
plt.show()

selector = SelectKBest(k=2)
# 选择相关性最大的两个feature
X_new = selector.fit_transform(X, y)

results = []
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_result = cross_val_score(model, X_new, y, cv=kfold)
    results.append((name, cv_result))
for i in range(len(results)):
    print("name: {}; cross val score: {}".format(
        results[i][0], results[i][1].mean()))

# 画出数据
plt.figure(figsize=(10, 6))
plt.ylabel("BMI")
plt.xlabel("Glucose")
plt.scatter(X_new[y==0][:, 0], X_new[y==0][:, 1], c='r', s=20, marker='o')       # 画出样本
plt.scatter(X_new[y==1][:, 0], X_new[y==1][:, 1], c='g', s=20, marker='^')
plt.show()
# 根据散点图的分类情况可以看出KNN方法的预测方法不好， 因为两类数据基本上都是重合的