import pandas as pd
import numpy as np
from scipy.optimize import linprog
from copy import copy

class OptimizationResult:
    def __init__(self):
        pass

def solve_lp(data=None, include = None, exclude = None, group_var = "group", group = "ovca", scale = False,
             x_sum = 1, id_var = "ID", eps = 1, my = 0.5):

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

    maxima = []
    minima = []
    if scale:
        for col in data.columns:
            if col in cols:
                max = np.max(data[col])
                min = np.min(data[col])
                data[col] = (data[col]-min)/(max - min)
                maxima.append(max)
                minima.append(min)
        def scaler():
            pass

    A_t = data[data[group_var] != group][[col for col in cols]].to_numpy()
    A_o = data[data[group_var] == group][[col for col in cols]].to_numpy()
    n_t = A_t.shape[0]
    n_o = A_o.shape[0]
    n_dim = A_o.shape[1]
    A = np.block([[A_t], [A_o]])

    A = np.block([np.ones((n_t + n_o, 1)), A])
    IDs = data[data[group_var] != group][id_var].to_numpy().tolist()
    IDs_healthy = copy(IDs)


    A_ub = np.block([
        [-np.ones((n_t, 1)), -A_t, -np.eye(n_t), np.zeros((n_t, n_o))], # healthy
        [np.ones((n_o, 1)), A_o, np.zeros((n_o, n_t)), -np.eye(n_o)], #np.zeros((n_o, n_dim))], # cancer
        #[np.zeros(1), np.zeros(1), np.mean(A_o, axis=0) - np.mean(A_t, axis=0), np.zeros(n_dim)], # feasibility condition, not needed :)
        #[np.zeros((n_dim, 1)), np.zeros((n_dim, 1)), np.eye(n_dim), -np.eye(n_dim)], # xi > x
        #[np.zeros((n_dim, 1)), np.zeros((n_dim, 1)), -np.eye(n_dim), -np.eye(n_dim)], # xi > -x
    ])

    A_eq = np.block([#[np.zeros(1), np.zeros(1)], #np.zeros(n_dim), np.ones(n_dim)],
                     [np.zeros(1), np.ones(n_dim), np.zeros(n_t + n_o)]])#, np.zeros(n_dim)]])  # |xi| == 1

    c = np.array((n_dim + 1) *[0] + n_t * [2*my] + n_o * [2-2*my])
    b_eq= np.array([-x_sum])
    b_ub = np.array(n_t * [-eps] + n_o * [-eps])
    bounds = np.array((n_dim +1) * [None, None] + (n_t + n_o) * [0, None]).reshape((1 + n_dim + n_t + n_o, 2))
    lin = linprog(c, A_ub=A_ub, b_ub=b_ub,
                   A_eq=None, b_eq=None,
                   bounds=bounds)

    zmax = np.min(lin.x[n_dim + 1: n_dim + n_o + 1])
    hyperplane = lin.x[:n_dim + 1]
    hyperplane[0] -= zmax + eps
    lin.hyperplane = hyperplane
    lin.A = A
    lin.names = ["intercept"] + cols

    y = lin.A.dot(lin.hyperplane)
    lin.y = y
    return lin


    # the following is not needed anymore
    positive = (lin.slack[n_t:n_t+n_o] > lin.fun)

    def step(lin, A, nt, IDs): # not needed

        positive = (lin.slack[nt:nt + n_o] > lin.fun)

        best_sol = None
        best_A = None

        b_ub = np.array((nt-1) * [0] + n_o * [0]) #+ 2 * n_dim * [0])

        z = lin.fun
        remove = None
        best_i = None
        for i in np.where(lin.slack[:nt] == 0)[0]:
            selector = list(range(A.shape[0]))
            selector.remove(i)

            A_step = A[selector, :]

            lin1 = linprog(c, A_ub=A_step, b_ub=b_ub,
                           A_eq=A_eq, b_eq=[-x_sum], #[x_constraint, -x_sum],
                           bounds=(None, None))
            lin2 = linprog(c, A_ub=A_step, b_ub=b_ub,
                           A_eq=A_eq, b_eq=[x_sum], #[x_constraint, x_sum],
                           bounds=(None, None))

            if sum(lin1.slack[nt-1:nt-1+n_o] > lin1.fun) > sum(positive):
                best_sol = lin1
                positive = (lin1.slack[nt-1:nt-1+n_o] > lin1.fun)
                best_A = A_step
                z = lin1.fun
                best_i = i

            if sum(lin2.slack[nt-1:nt-1+n_o] > lin2.fun) > sum(positive):
                best_sol = lin2
                positive = (lin2.slack[nt-1:nt-1+n_o] > lin2.fun)
                best_A = A_step
                z = lin2.fun
                best_i = i

        if best_sol:
            remove = IDs.pop(best_i)
            return best_sol, best_A, nt -1, positive, IDs, remove

        for i in np.where(lin.slack[:nt] == 0)[0]:
            selector = list(range(A.shape[0]))
            selector.remove(i)
            A_step = A[selector, :]

            lin1 = linprog(c, A_ub=A_step, b_ub=b_ub,
                           A_eq=A_eq, b_eq=[-x_sum], #[x_constraint, -x_sum],
                           bounds=(None, None))
            lin2 = linprog(c, A_ub=A_step, b_ub=b_ub,
                           A_eq=A_eq, b_eq=[x_sum], #[x_constraint, x_sum],
                           bounds=(None, None))

            if lin1.fun < z:
                best_sol = lin1
                positive = (lin1.slack[nt-1:nt-1+n_o] > lin1.fun)
                best_A = A_step
                z = lin1.fun
                best_i = i

            if lin2.fun < z:
                best_sol = lin2
                positive = (lin2.slack[nt-1:nt-1+n_o] > lin2.fun)
                best_A = A_step
                z = lin2.fun
                best_i = i

        remove = IDs.pop(best_i)

        return best_sol, best_A, nt -1, positive, IDs, remove

    solutions = [lin.x]
    As = [A_ub]
    positives = [positive]
    removes = ["None"]
    results = [lin]
    i = 0
    while lin.fun > 0:
        lin, A_ub, n_t, positive, IDs, remove = step(lin, A_ub, n_t, IDs)
        print("step ", i+1)
        i += 1
        solutions.append(lin.x)
        results.append(lin)
        As.append(A_ub)
        positives.append(positive)
        removes.append(remove)


    opt = OptimizationResult()
    solutions = pd.DataFrame({"step" + str(nr) + "removed" + str(remove): sol for nr, (remove, sol) in enumerate(zip(removes, solutions))}, index = ["z", "intercept"] + ["x_" + col for col in cols])# + ["xi_" + col for col in cols])
    classifiers = {group_var: data.group}

    classifiers.update(
        {"step" + str(nr) + "removed" + str(remove): [(ID in removes[:nr]) for ID in IDs_healthy] + positives[nr].tolist() for
         nr, remove in enumerate(removes)})

    classifiers = pd.DataFrame(classifiers)

    opt.solutions = solutions
    opt.classifiers = classifiers
    opt.maxima = maxima
    opt.minima = minima
    opt.data = A

    return opt