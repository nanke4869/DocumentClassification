#-*-coding:ISO-8859-1-*-

from numpy import *
import numpy as np


# ����洢��������
class opStruct():
    def __init__(self, dataMatIn, classLabels, C, toler, kTup, kTup1):
        self.X = dataMatIn  # ����
        self.labelMat = classLabels  # ��ǩ
        self.C = C  # ���̶�
        self.toler = toler  # �������̶�
        self.m = shape(dataMatIn)[0]  # ���ݵĸ���
        self.alphas = mat(zeros((self.m, 1)))  # alpha ֵ��ÿ�����ݶ�Ӧһ��alpha
        self.b = 0  # ������
        self.eCache = mat(zeros((self.m, 2)))  # ���������±�
        self.k = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.k[:, i] = kernelTrans(self.X, self.X[i, :], kTup, kTup1)


def getrightfile(filename1, filename2):
    """
    ��������
    """
    a = np.load(filename1)
    b = np.load(filename2)
    a_float = map(float, a)
    b_float = map(float, b)
    dataMatIn = a_float.tolist()
    classLabels = b_float.tolist()
    return dataMatIn, classLabels


def kernelTrans(X, A, kTup, kTup1):
    """
    ����˺���
    """
    m, n = shape(X)
    k = mat(zeros((m, 1)))
    if kTup == 'lin':
        k = X * A.T
    elif kTup == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A  # ÿһ�м�ȥA�����Լ���
            k[j] = deltaRow * deltaRow.T
        k = exp(k / (-1 * kTup1 ** 2))  # �������õĹ�ʽ
    return k


def clipAlpha(ai, H, L):
    """
    ��֤alpha�����ڷ�Χ��
    """
    if ai > H:
        ai = H
    elif ai < L:
        ai = L
    return ai


def selectJrand(i, oS):
    """
    ���ѡ��ڶ�����ͬ��alpha
    """
    j = i
    while i == j:
        j = int(np.random.uniform(0, oS.m))
    return j


def calcEk(oS, k):
    """
    �������
    """
    fXk = float((multiply(oS.alphas, oS.labelMat)).T * oS.k[:, k] + oS.b)  # Ԥ��ֵ
    Ek = fXk - oS.labelMat[k]  # ���ֵ
    return Ek


def selectJ(i, oS, Ei):
    """
    ѡ��ڶ���alpha �����������
    """
    maxK = -1
    maxDelaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcaheList = nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcaheList) > 0:
        for k in validEcaheList:
            if k == i:  # ȡ��ͬ�� alpha
                continue
            Ek = calcEk(oS, k)  # ����k�����������ʵֵ֮������
            deltaE = abs(Ei - Ek)  # ����Ei ������Զ��
            if maxDelaE < deltaE:
                maxDelaE = deltaE
                maxK = k  # ��Ei�������K
                Ej = Ek  # K�����
        return maxK, Ej
    else:
        j = selectJrand(i, oS)
        Ej = calcEk(oS, j)  # ����Ԥ��ֵ����ʵֵ�����
    return j, Ej


def updateEk(oS, k):
    """
    �������
    """
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    """
    SMO �Ż�
    """
    Ei = calcEk(oS, i)
    # ���������ķ�Χ�⣬���С�ڹ涨�����Ͳ���Ҫ������
    if ((oS.labelMat[i] * Ei) <= oS.toler and oS.alphas[i] <= oS.C) or \
            ((oS.labelMat[i] * Ei) <= oS.toler and oS.alphas[i] >= 0):
        j, Ej = selectJ(i, oS, Ei)  # ѡ����һ��alphaj��Ԥ��ֵ����ʵֵ�Ĳ�
        alphaIold = oS.alphas[i].copy()  # ����alpha����Ϊ��߻��õ�
        alphaJold = oS.alphas[j].copy()

        if (oS.labelMat[i] != oS.labelMat[j]):  # �������һ�� һ������ һ������
            L = max(0, oS.labelMat[j] - oS.labelMat[i])  # Լ������ ��������
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])

        if L == H:
            print('L == H')
            return 0

        # ���ú˺���
        eta = 2.0 * oS.k[i, j] - oS.k[i, i] - oS.k[j, j]
        if eta > 0:
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta  # ���ǰ����Ĺ�ʽ���
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)  # ��L��H��Χ��
        updateEk(oS, j)

        if (oS.alphas[j] - alphaJold) < 0.0001:
            return 0

        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)

        # ���ú˺�����
        b1 = oS.b - Ei - oS.labelMat[i] * oS.k[i, i] * (oS.alphas[i] - alphaIold) - oS.labelMat[j] * oS.k[i, j] * (
                    oS.alphas[j] - alphaJold)
        b2 = oS.b - Ej - oS.labelMat[i] * oS.k[i, j] * (oS.alphas[i] - alphaIold) - oS.labelMat[j] * oS.k[j, j] * (
                    oS.alphas[j] - alphaJold)

        # ����b
        if oS.alphas[i] < oS.C and oS.alphas[i] > 0:
            oS.b = b1
        elif oS.alphas[j] < oS.C and oS.alphas[j] > 0:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def calcWs(alpha, dataArr, classLabels):
    """
    ����alpha ��÷����Ȩ������
    :param alpha:
    :param dataArr: ѵ������
    :param classLabels: ѵ����ǩ
    :return:
    """
    X = mat(dataArr)
    labelMat = mat(classLabels).T  # ���������
    m, n = shape(X)
    w = zeros((n, 1))  # w�ĸ����� ���ݵ�ά��һ��
    for i in range(m):
        w += multiply(alpha[i] * labelMat[i], X[i, :].T)  # alpha[i] * labelMat[i]����һ������  X[i,:]ÿ���У������ݣ���ΪwΪ��������������Ҫת��
    return w


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup='lin', kTup1=0):
    """
    ����������
    :param dataMatIn: ѵ������
    :param classLabels: ѵ����ǩ
    :param C: ����
    :param toler:�ݴ��
    :param maxIter: ����������
    :param kTup: �˺������Ͳ���
    :param kTup1: �˺������Ͳ���
    :return:
    """
    oS = opStruct(mat(dataMatIn), mat(classLabels).T, C, toler, kTup, kTup1)
    iter = 0
    entireSet = True
    alphaPairedChanged = 0
    while (iter < maxIter) and ((alphaPairedChanged > 0) or (entireSet)):
        alphaPairedChanged = 0
        if entireSet:
            # �������е����� ���и���
            for i in range(oS.m):
                alphaPairedChanged += innerL(i, oS)
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < oS.C))[0]
            for i in nonBoundIs:
                alphaPairedChanged += innerL(i, oS)
            iter += 1

        if entireSet:
            entireSet = False
        elif (alphaPairedChanged == 0):
            entireSet = True
    return oS.b, oS.alphas


if __name__ == '__main__':
    dataMatIn, classLabels = getrightfile('email_smo.npy', 'label_smo.npy')
    b, alphas = smoP(dataMatIn, classLabels, C=0.6, toler=0.001, maxIter=40, kTup='lin', kTup1=1)
    print("b:\n{}".format(b))
    print("alphas:\n{}".format(alphas))