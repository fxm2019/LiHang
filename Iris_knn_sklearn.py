import numpy as np
from sklearn.datasets import load_iris
#from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

# iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
n = 0

# print(iris_X[:2, :])
# print(iris_Y)

X_train,X_test,y_train,y_test = train_test_split(iris_X, iris_y, test_size=0.3)
print(y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print(knn.predict(X_test))
print(y_test)

y_predict = knn.predict(X_test)
for i in range(y_test.size):
    if y_predict[i]!=y_test[i]:
        n = n + 1

Accuracy_rate = 100*(y_test.size-n)/y_test.size
print("Accuracy rate is {:.2f}% !".format(Accuracy_rate))
