import pandas as pd
import numpy as np
from cvxopt.solvers import sdp, qp
from cvxopt import matrix
from scipy.linalg import sqrtm
from itertools import combinations as comb


def mycomb(l, k):
    return list(comb(l, int(k)))

class OptimizationResult:
    def __init__(self):
        pass

def solve_ellipse(data=None, group_var = "group", group = "ovca", id_var = "ID", eps = 0.0001,
                  include = None, exclude = "OP", checklinear = False, mu = 0.5,
                  nopos = False, perfect = False):

    if data is None:
        data = pd.read_csv("Data_Klopf.csv", sep=";")
    else:
        data = pd.DataFrame(data)
    if include is not None:
        cols = [col for col in data.columns if col in include and col != group_var and col != id_var]
    else:
        cols = [col for col in data.columns if col != group_var and col != id_var]
    if exclude is not None:
        if not isinstance(exclude, list):
            exclude = [exclude]
        cols = [col for col in cols if all(str(ex) not in col for ex in exclude)]

    newcols = list(cols)

    for col1 in cols:
        newcol = False
        for col2 in cols:
            if col1 == col2:
                newcol = True
                data[str(col1) + str(col2)] = data[col1]**2
                newcols.append(str(col1) + str(col2))
                continue
            if newcol:
                data[str(col1) + str(col2)] = 2*data[col1]*data[col2]
                newcols.append(str(col1) + str(col2))
    sick = data[data[group_var] == group][[col for col in newcols]].to_numpy()
    healthy = data[data[group_var] != group][[col for col in newcols]].to_numpy()

    ndim = len(cols)
    ndimsqh = int(ndim * (ndim + 1) / 2)
    no = sick.shape[0]
    nt = healthy.shape[0]
    c = nt * [1-mu] + no * [mu] + (1 + ndim + ndimsqh) * [0]
    c = matrix(c, tc="d")

    G0 = np.block([
        [-np.eye(no + nt), np.zeros((no + nt, 1 + ndim + ndimsqh))], # z > 0
        [-np.eye(nt), np.zeros((nt, no)), -np.ones((nt, 1)), -healthy], #  separating healthy
        [np.zeros((no, nt)), -np.eye(no), np.ones((no, 1)), sick],  # from sick
    ])

    G0 = matrix(G0, tc = "d")
    h0 = np.array((nt + no) * [0] + nt * [-eps] + no * [-eps])

    h0 = matrix(h0, tc = "d")

    if perfect:
        c = [-1] + (nt + no) * [0] + + (1 + ndim + ndimsqh) * [0]
        c = matrix(c, tc="d")
        G0 = np.block([
            [np.ones((no+nt, 1)), -np.eye(no + nt), np.zeros((no + nt, 1 + ndim + ndimsqh))],  # z0 < z
            [np.zeros((nt, 1)), np.zeros((nt, nt)), np.zeros((nt, no)), -np.ones((nt, 1)), -healthy],  # separating healthy
            [np.zeros((no, 1)), np.zeros((no, nt)), np.eye(no), np.ones((no, 1)), sick],  # from sick
        ])
        G0 = matrix(G0, tc="d")

    def createG1(perfect):
        if perfect:
            add = 1
        else:
            add = 0
        vecs1 = ndim**2 * (nt+no+ndim + 1 + add) * [0]
        vecs2 = []
        for i in range(ndim):
            for j in range(i, ndim):
                vec = ndim**2 * [0]
                vec[i*ndim + j] = -1
                vec[j*ndim + i] = -1
                vecs2 = vecs2 + vec
        return np.array(vecs1 + vecs2).reshape((nt + no + ndim + 1 + add + ndimsqh, ndim**2)).transpose()

    G1 = matrix(createG1(perfect), tc = "d")

    G2 = np.block([
        [np.zeros((ndim**2, no + nt + ndim + 1)), np.eye(ndim**2)],
    ])
    G2 = matrix(G2, tc = "d")
    h1 = np.zeros((ndim, ndim))
    h1 = matrix(h1, tc = "d")
    #np.savetxt("Ellipsoid/G0.csv", np.array(G0), delimiter=";")
    #np.savetxt("Ellipsoid/G1.csv", np.array(G1), delimiter=";")
    #np.savetxt("Ellipsoid/h0.csv", np.array(h0), delimiter=";")
    #np.savetxt("Ellipsoid/h1.csv", np.array(h1), delimiter=";")
    #np.savetxt("Ellipsoid/c.csv", np.array(c), delimiter=";")

    if checklinear:
        sol = sdp(c, Gl=G0, hl=h0, Gs=[G1, -G1], hs=[h1, h1], solver="dsdp")
    elif nopos:
        sol = sdp(c, Gl=G0, hl=h0, solver="dsdp")
    else:
        sol = sdp(c, Gl = G0, hl = h0, Gs = [G1], hs = [-h1], solver = "dsdp")

    #np.savetxt("Ellipsoid/x.csv", np.array(sol["x"]), delimiter=";")

    IDs = data[data[group_var] != group][id_var].to_numpy().tolist()
    IDs = IDs + data[data[group_var] == group][id_var].to_numpy().tolist()

    sol["IDs"] = IDs
    sol["data"] = np.block([[np.ones((nt, 1)), healthy], [np.ones((no, 1)), sick]])
    sol["solution"] = np.array(sol["x"])[nt + no:, ]
    sol["x"] = np.array(sol["x"])
    matvals = np.array(sol["x"][no+nt+ndim+1:]).tolist()
    if perfect:
        sol["solution"] = sol["solution"][1:]
        sol["x"] = sol["x"][1:]
        matvals =  matvals[1:]
    mat = ndim**2 * [0]

    for i in range(ndim):
        for j in range(i, ndim):
            val = matvals.pop(0)
            mat[i*ndim + j] = val
            mat[j*ndim + i] = val

    A = np.array(mat).reshape((ndim, ndim))
    b = np.array(sol["x"][no + nt + 1:no + nt + ndim + 1])
    sol["b_orig"] = b
    c = np.array(sol["x"][no + nt])
    try:
        x0 = -np.linalg.solve(A, b/2)
        w = -1/2*b.transpose().dot(x0) - c
        A = A/w
        b = b/w
        c = c/w
    except:
        x0 = "A singular"
        w = "A singular"
    eig = np.linalg.eigh(A)

    sol["A"] = A
    sol["b"] = b
    sol["c"] = c
    sol["x0"] = x0
    sol["w"] = w
    sol["eig"] = eig
    sol["names"] = ["ones"] + newcols

    return sol


def min_eigen(A, alpha = 100):
    eig = np.linalg.eigh(A)
    mymin = max(eig.eigenvalues/alpha)
    return eig.eigenvectors.dot(np.diag( [max(val, mymin) for val in eig.eigenvalues] ) ).dot(eig.eigenvectors.transpose())


def sqrt_project(A, index1=1, index2=2):
    solve = np.linalg.inv
    A = np.array(A)
    ndim = A.shape[0]
    if ndim == 2:
        eig = np.linalg.eigh(A)
        return eig.eigenvectors.dot(np.diag([1/np.sqrt(val) for val in eig.eigenvalues])).dot(
            eig.eigenvectors.transpose())

    indz = [ind for ind in range(ndim) if ind != index1 -1 and ind != index2 -1]
    indx = [index1-1, index2-1]

    Ax = A[np.ix_(indx, indx)]
    Az = A[np.ix_(indz, indz)]

    Axz = A[np.ix_(indx, indz)]

    eigz = np.linalg.eigh(Az)
    Az_1 = eigz.eigenvectors.dot(np.diag([1/val for val in eigz.eigenvalues])).dot(eigz.eigenvectors.transpose())
    A0 = Ax - Axz.dot(Az_1).dot(Axz.transpose())

    eig = np.linalg.eigh(A0)

    return eig.eigenvectors.dot(np.diag([1/np.sqrt(val) for val in eig.eigenvalues])).dot(eig.eigenvectors.transpose())

def project(A, index1=1, index2=2):
    solve = np.linalg.inv
    A = np.array(A)
    ndim = A.shape[0]
    if ndim == 2:
        eig = np.linalg.eigh(A)
        return eig.eigenvectors.dot(np.diag([1/np.sqrt(val) for val in eig.eigenvalues])).dot(
            eig.eigenvectors.transpose())

    indz = [ind for ind in range(ndim) if ind != index1 -1 and ind != index2 -1]
    indx = [index1-1, index2-1]

    Ax = A[np.ix_(indx, indx)]
    Az = A[np.ix_(indz, indz)]

    Axz = A[np.ix_(indx, indz)]

    eigz = np.linalg.eigh(Az)
    Az_1 = eigz.eigenvectors.dot(np.diag([1/val for val in eigz.eigenvalues])).dot(eigz.eigenvectors.transpose())
    A0 = Ax - Axz.dot(Az_1).dot(Axz.transpose())

    return A0


def quadp(A, x0, x, index1, index2):
    A = np.array(A)

    x0 = np.array(x0).reshape(x0.shape[0])
    x = np.array(x)
    index1 = int(index1)
    index2 = int(index2)
    ndim = A.shape[0]
    indx = [index1-1, index2-1]
    indz = [ind for ind in range(ndim) if ind != index1 - 1 and ind != index2 - 1]

    import pdb
    pdb.set_trace()

    c = 1 - (x - x0[np.ix_(indx)]).transpose().dot(A[np.ix_(indx, indx)]).dot(x - x0[np.ix_(indx)])
    Q = matrix(A[np.ix_(indz, indz)], tc="d")
    r = matrix(2*(x-x0[np.ix_(indx)]).transpose().dot(A[np.ix_(indx, indz)]).transpose(), tc="d")

    sol = qp(Q, r)
    return sol["primal objective"] - c






