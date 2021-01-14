import time
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import metrics


if __name__ == '__main__':

    data = pd.read_table('preprocess_email.csv', header=0, encoding='ISO-8859-1', sep=',', index_col=0)
    data = data.as_matrix()[1:][1:]
    X, y = data[:, 0:-1], data[:, -1].astype(int)

    kf = KFold(n_splits=5, shuffle=True)

    P, R, F1 = [], [], []
    k = 1

    time_start = time.time()
    print('----------开始五折SVM训练----------')

    for train_index, test_index in kf.split(X):
        time_kfold_start = time.time()
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

        # 建立模型
        model = svm.SVC(kernel='rbf', gamma='scale', cache_size=800)

        # 模型拟合
        model.fit(X_train, y_train)

        # 模型预测
        y_predict = model.predict(X_test)

        # 评价指标
        precision = metrics.precision_score(y_test, y_predict)
        recall = metrics.recall_score(y_test, y_predict)
        f1 = metrics.f1_score(y_test, y_predict)
        P.append(precision)
        R.append(recall)
        F1.append(f1)

        time_kfold_end = time.time()
        print("KFold_" + str(k) + "===>")
        print("耗时：" + str(time_kfold_end - time_kfold_start) + "s")
        print("precision: " + str(precision))
        print("recall: " + str(recall))
        print("f1: " + str(f1))
        k += 1

    time_end = time.time()
    print("----------五折SVM训练完成----------")
    print("全部耗时：" + str(time_end - time_start) + "s")
    print("average precision: " + str(np.mean(P)))
    print("average recall: " + str(np.mean(R)))
    print("average f1: " + str(np.mean(F1)))