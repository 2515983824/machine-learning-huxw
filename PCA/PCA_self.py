import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# K近邻分类器
from sklearn.neighbors import KNeighborsClassifier
# PCA
from sklearn.decomposition import PCA

if __name__ == '__main__':
    digits = datasets.load_digits()
    x = digits.data
    y = digits.target
    print(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=666)

    # k近邻
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(x_train, y_train)
    acc = knn_clf.score(x_test, y_test)
    print("knn acc is %s: " % acc)

    # PCA
    pca = PCA(n_components=2)
    pca.fit(x_train, y_train)
    x_train_reduction = pca.transform(x_train)
    x_test_reduction = pca.transform(x_test)

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(x_train_reduction, y_train)
    acc = knn_clf.score(x_test_reduction, y_test)
    print("After PCA, knn acc is %s: " % acc)

    # 显示方差的改变曲线
    plt.plot([i for i in range(x_train.shape[1])],
             [np.sum(pca.explained_variance_ratio_[:i + 1]) for i in range(x_train.shape[1])])
    plt.show()

    # 降维可视化
    pca = PCA(n_components=2)
    x_reduction = pca.fit_transform(x)
    # print(x_reduction)

    for i in range(10):
        plt.scatter(x_reduction[y == i, 0], x_reduction[y == i, 1], alpha=0.8)
        # print(x_reduction[y == i, 0], x_reduction[y == i, 1])
    plt.show()
