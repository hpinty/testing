"""
main function from IJCAI23-Label generation with consistency on the graph for multi-label feature selection.
input: matrices and hyper-paras is must.
output: the no. of selected features.
"""

import numpy as np
from numpy.random import seed
from numpy import linalg as LA
from scipy import linalg
from scipy.spatial import distance
from skfeature.utility.construct_W import construct_W
import skfeature.utility.entropy_estimators as ees
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
import pylab as pl

eps = 2.2204e-16

"""
function: search the range for discrete label matrix
"""
def search_range(y11):
    y11_dict = {}
    avg_dur = 0
    for p in range(len(y11)):
        if np.around(y11[p]) not in y11_dict.keys():
            y11_dict[np.around(y11[p])] = p
    y11_keys = list(y11_dict.keys())
    # 平均组间隔或者中位数组间隔：
    for k in range(len(y11_keys)-1):
        avg_dur += y11_dict[y11_keys[k + 1]] - 1 - y11_dict[y11_keys[k]] + 1
        y11_dict[y11_keys[k]] = (y11_dict[y11_keys[k]],y11_dict[y11_keys[k+1]]-1)
    avg_dur += len(y11) - 1 - y11_dict[y11_keys[k + 1]] + 1
    y11_dict[y11_keys[k+1]] = (y11_dict[y11_keys[k+1]],len(y11)-1)

    avg_dur = int(avg_dur / len(y11_keys))
    # 平均组间隔或者中位数组间隔：

    return y11_dict,avg_dur

"""
function: generation process
"""
def flip_function(Y, YY, U,idx4, idx2, idx5, dump2, dump5, y_miss_loc, gamma,options):
    u_1 = idx4[0]
    y_1 = idx2[0].tolist()
    y_l = idx5[0].tolist()
    y11 = dump2[0].tolist()
    yll = dump5[0].tolist()
    # 范围初始化：
    # 前后查找范围
    y11_dict, avg1 = search_range(y11)
    yll_dict, avg2 = search_range(yll)
    y_0 = Y[0][y_miss_loc]

    new_y_miss = Y[:, y_miss_loc:y_miss_loc + 1]
    new_y_miss[0] = y_0
    for uu in range(1, len(u_1)):

        yy = y_1.index(u_1[uu])
        value_y = y11[yy]
        range_y11 = y11_dict[np.around(value_y)]
        yy2 = y_l.index(u_1[uu])
        value_yl = yll[yy2]
        range_yll = yll_dict[np.around(value_yl)]

        if uu > range_yll[1] and range_y11[1] < uu:
            YY[u_1[uu]][y_miss_loc] = y_0
        elif uu > range_yll[1] and range_y11[0] > uu:
            YY[u_1[uu]][y_miss_loc] = (~y_0.astype('bool')).astype(int)

        if uu < range_yll[0] and range_y11[1] < uu:
            YY[u_1[uu]][y_miss_loc] = (~y_0.astype('bool')).astype(int)
        elif uu < range_yll[0] and range_y11[0] > uu:
            YY[u_1[uu]][y_miss_loc] = y_0

        if uu >= range_yll[0] and uu <= range_yll[1] and (uu < range_y11[0] or uu > range_y11[1]):
            YY[u_1[uu]][y_miss_loc] = y_0

    # 求yy的lap：
    Syyy = construct_W(YY, **options)
    Syyy = Syyy.A
    Ayyy = np.diag(np.sum(Syyy, 0))
    Lyyy = Ayyy - Syyy

    part11 = (1 - gamma) * np.trace(np.dot(np.dot(U.T, Lyyy), U))

    obj2_1 = part11

    YY1 = YY.copy()
    # '0' 的原位置变为取反，再做这个操作：
    temp = ~Y[0][y_miss_loc].astype("bool")
    y_0 = temp.astype(int)
    new_y_miss = Y[:, y_miss_loc:y_miss_loc + 1]
    new_y_miss[0] = y_0
    for uu in range(1, len(u_1)):

        yy = y_1.index(u_1[uu])
        value_y = y11[yy]
        range_y11 = y11_dict[np.around(value_y)]
        yy2 = y_l.index(u_1[uu])
        value_yl = yll[yy2]
        range_yll = yll_dict[np.around(value_yl)]

        if uu > range_yll[1] and range_y11[1] < uu:
            YY[u_1[uu]][y_miss_loc] = y_0
        elif uu > range_yll[1] and range_y11[0] > uu:
            YY[u_1[uu]][y_miss_loc] = (~y_0.astype('bool')).astype(int)

        if uu < range_yll[0] and range_y11[1] < uu:
            YY[u_1[uu]][y_miss_loc] = (~y_0.astype('bool')).astype(int)
        elif uu < range_yll[0] and range_y11[0] > uu:
            YY[u_1[uu]][y_miss_loc] = y_0

        if uu >= range_yll[0] and uu <= range_yll[1] and (uu < range_y11[0] or uu > range_y11[1]):
            YY[u_1[uu]][y_miss_loc] = y_0

    YY2 = YY.copy()

    # 求yy的lap：
    Syyy = construct_W(YY, **options)
    Syyy = Syyy.A
    Ayyy = np.diag(np.sum(Syyy, 0))
    Lyyy = Ayyy - Syyy

    part12 = (1 - gamma) * np.trace(np.dot(np.dot(U.T, Lyyy), U))

    obj2_2 = part12

    return obj2_1, obj2_2, YY1, YY2


def LGCM(X, Y, y_miss_loc, select_nub, gamma,theta):
    # initialization
    num, dim = X.shape
    num, label_num = Y.shape

    seed(4)
    Y_ori = Y.copy()
    y_miss_loc = y_miss_loc
    # inject the random value
    y_miss = np.random.randint(0, 2, size=num)
    Y[:, y_miss_loc:y_miss_loc + 1] = y_miss.reshape(num, 1)
    # check the precision for the random label
    ori_y_miss = Y_ori[:, y_miss_loc:y_miss_loc + 1]
    y_score = accuracy_score(ori_y_miss, y_miss)
    Y_less = np.delete(Y_ori, y_miss_loc, axis=1)

    options = {'metric': 'euclidean', 'neighbor_mode': 'knn', 'k': 5, 'weight_mode': 'heat_kernel', 't': 1.0}
    Sx = construct_W(X, **options)
    Sx = Sx.A
    D = pairwise_distances(X)
    D **= 2
    # sort the distance matrix D in ascending order
    Ax = np.diag(np.sum(Sx, 0))
    Lx = Ax - Sx

    D = pairwise_distances(Y)
    D **= 2
    # sort the distance matrix D in ascending order
    dump2 = np.sort(D, axis=1)
    idx2 = np.argsort(D, axis=1)

    D = pairwise_distances(Y_less)
    D **= 2
    # sort the distance matrix D in ascending order
    dump5 = np.sort(D, axis=1)
    idx5 = np.argsort(D, axis=1)

    k = 10
    U = np.random.rand(num, k)
    M = np.random.rand(k, dim)

    iter = 0
    obj = []
    obji = 1
    obji2 = 1
    cver_lst=[]
    # updating process
    while 1:
        D = pairwise_distances(U)
        D **= 2
        # sort the distance matrix D in ascending order
        idx4 = np.argsort(D, axis=1)
        # the ground-truth index we choose is zero
        YY = Y.copy()
        obj2_1, obj2_2, YY1, YY2 = flip_function(Y, YY, U,idx4, idx2, idx5, dump2, dump5, y_miss_loc, gamma,options)

        if obj2_2 > obj2_1:
            Y = YY1.copy()
        else:
            Y = YY2.copy()

        Snewy = construct_W(Y, **options)
        Snewy = Snewy.A
        Anewy = np.diag(np.sum(Snewy, 0))
        Lnewy = Anewy - Snewy

        Btmp = np.sqrt(np.sum(np.multiply(M.T, M.T), 1) + eps)
        d1 = 0.5 / Btmp
        D = np.diag(d1.flat)

        U = np.multiply(U, np.true_divide(np.dot(X, M.T) + gamma * np.dot(Sx, U) +(1-gamma)*np.dot(Snewy, U),
                                          np.dot(np.dot(U, M), M.T) + gamma * np.dot(Ax, U) + (1-gamma)*np.dot(Anewy, U)+eps))
        M = np.multiply(M, np.true_divide(np.dot(U.T, X),
                                          np.dot(np.dot(U.T, U), M) +
                                          theta * np.dot(M, D) + eps))

        part1 = (1-gamma) * np.trace(np.dot(np.dot(U.T, Lnewy), U))
        part4 = 2 * theta * np.trace(np.dot(np.dot(M, D), M.T))
        part5 = pow(LA.norm(X - np.dot(U, M), 'fro'), 2)
        part3 = gamma * np.trace(np.dot(np.dot(U.T, Lx), U))

        objectives = part3 + part5 + part1  + part4
        objectives2 = part1
        obj.append(objectives)
        cver = abs((objectives - obji) / float(obji))
        cver_lst.append(cver)
        obji = objectives

        cver2 = abs((objectives2 - obji2) / float(obji2))
        obji2 = objectives2

        iter = iter + 1
        if (iter > 2 and (cver < 0.005 or iter == 100) and (cver2 < 0.0002 or iter == 100)):
            break

    # Final process
    D = pairwise_distances(U)
    D **= 2
    # sort the distance matrix D in ascending order
    idx4 = np.argsort(D, axis=1)

    YY = Y.copy()
    obj2_1, obj2_2, YY1, YY2 = flip_function(Y, YY, U,idx4, idx2, idx5, dump2, dump5, y_miss_loc, gamma,options)
    if obj2_2 > obj2_1:
        new_y_miss = YY1[:, y_miss_loc:y_miss_loc + 1]
    else:
        new_y_miss = YY2[:, y_miss_loc:y_miss_loc + 1]
    # record iterations:
    obj_value = np.array(obj)
    obj_function_value = []
    for i in range(iter):
        temp_value = float(obj_value[i])
        obj_function_value.append(temp_value)
    # sorting the features
    score = np.sum(np.multiply(M.T, M.T), 1)
    idx = np.argsort(-score, axis=0)
    idx = idx.T.tolist()
    l = [i for i in idx]
    n = 1
    F = [l[i:i + n] for i in range(0, len(l), n)]
    F = np.matrix(F)
    # recording the F_value
    ll = [i for i in obj_function_value]
    n = 1
    F_value = [ll[i:i + n] for i in range(0, len(ll), n)]
    F_value = np.matrix(F_value)

    y_final_score = 0 # try to add what you want
    # except features, the other paras could be selected to return
    return F[0:select_nub, :], F_value[:, :], iter, y_final_score, y_score, new_y_miss
