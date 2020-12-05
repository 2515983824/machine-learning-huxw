import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report


def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    # 模型预测
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)


def plot_svc_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    # 模型预测
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)

    w = model.coef_[0]
    b = model.intercept_[0]

    # w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    plot_x = np.linspace(axis[0], axis[1], 200)
    up_y = -w[0] / w[1] * plot_x - b / w[1] + 1 / w[1]
    down_y = -w[0] / w[1] * plot_x - b / w[1] - 1 / w[1]

    up_index = (up_y >= axis[2]) & (up_y <= axis[3])
    down_index = (down_y >= axis[2]) & (down_y <= axis[3])
    plt.plot(plot_x[up_index], up_y[up_index], color='black')
    plt.plot(plot_x[down_index], down_y[down_index], color='black')


# ----------------------- 线形SVM的实现 ------------------------
if __name__ == '__main__':
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    x = x[y < 2, :2]
    y = y[y < 2]

    # 数据标准化处理: 标准正态分布（高斯分布）
    standardScaler = StandardScaler()
    standardScaler.fit(x)
    x_standard = standardScaler.transform(x)

    # SVC训练
    svc = LinearSVC(C=1e9)
    svc.fit(x_standard, y)

    # 预测并显示
    plot_decision_boundary(svc, axis=[-3, 3, -3, 3])
    plot_svc_decision_boundary(svc, axis=[-3, 3, -3, 3])
    plt.scatter(x_standard[y == 0, 0], x_standard[y == 0, 1])
    plt.scatter(x_standard[y == 1, 0], x_standard[y == 1, 1])
    plt.show()




# ----------------------- 非线形SVM的实现：RBF核 ------------------------
if __name__ == '__main__':
    x, y = datasets.make_moons(noise=0.15, random_state=666)

    plt.scatter(x[y == 0, 0], x[y == 0, 1])
    plt.scatter(x[y == 1, 0], x[y == 1, 1])
    plt.show()


    # 我们在我们的pipline中首先进行标准化，然后我们实例化一个SVC，并且指定它的核函数为rbf以及它需要的一个超参数
    def RBFKernelSVC(gamma):
        return Pipeline([
            ("std_scaler", StandardScaler()),
            ("svc", SVC(kernel='rbf', gamma=gamma))
        ])


    svc = RBFKernelSVC(gamma=1)
    svc.fit(x, y)
    plot_decision_boundary(svc, axis=[-1.5, 2.5, -1.0, 1.5])
    plt.scatter(x[y == 0, 0], x[y == 0, 1])
    plt.scatter(x[y == 1, 0], x[y == 1, 1])
    plt.show()

    y_predict = svc.predict(x)
    print(classification_report(y, y_predict))
