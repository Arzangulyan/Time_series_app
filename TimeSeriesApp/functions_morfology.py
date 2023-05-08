import numpy as np
import cvxpy as cp


def convex_constraints_matrix(form_parameters):
    size = form_parameters[-1]
    for col_ind in range(len(form_parameters) - 1):
        if col_ind % 2 == 0:
            if form_parameters[col_ind] == 0:
                A = np.hstack([[-1, 1], np.zeros(size - 2)])
                for j in range(form_parameters[1] - 2):
                    m = [-0.5, 1, -0.5]
                    tmp = np.hstack([np.zeros(j), m, np.zeros(size - j - 3)])
                    A = np.vstack([A, tmp])
            else:
                m = [-1, 1]
                tmp = np.hstack(
                    [
                        np.zeros(form_parameters[col_ind] - 1),
                        m,
                        np.zeros(size - form_parameters[col_ind] - 1),
                    ]
                )
                A = np.vstack([A, tmp])
                tmp = np.hstack(
                    [
                        np.zeros(form_parameters[col_ind]),
                        m,
                        np.zeros(size - form_parameters[col_ind] - 2),
                    ]
                )
                A = np.vstack([A, tmp])
                for j in range(
                    form_parameters[col_ind], form_parameters[col_ind + 1] - 2
                ):
                    m = [-0.5, 1, -0.5]
                    tmp = np.hstack([np.zeros(j), m, np.zeros(size - j - 3)])
                    A = np.vstack([A, tmp])
        else:
            m = [1, -1]
            tmp = np.hstack(
                [
                    np.zeros(form_parameters[col_ind] - 1),
                    m,
                    np.zeros(size - form_parameters[col_ind] - 1),
                ]
            )
            A = np.vstack([A, tmp])
            tmp = np.hstack(
                [
                    np.zeros(form_parameters[col_ind]),
                    m,
                    np.zeros(size - form_parameters[col_ind] - 2),
                ]
            )
            A = np.vstack([A, tmp])
            for j in range(form_parameters[col_ind], form_parameters[col_ind + 1] - 2):
                m = [0.5, -1, 0.5]
                tmp = np.hstack([np.zeros(j), m, np.zeros(size - j - 3)])
                A = np.vstack([A, tmp])
    return A


def monotonous_constraints_matrix(form_parameters):
    size = form_parameters[-1]
    A = np.zeros(size)
    for col_ind in range(len(form_parameters) - 2):
        if col_ind % 2 == 0:
            m = [-1, 1]
            for j in range(form_parameters[col_ind], form_parameters[col_ind + 1]):
                tmp = np.hstack([np.zeros(j), m, np.zeros(size - j - 2)])
                A = np.vstack([A, tmp])
        else:
            m = [1, -1]
            for j in range(form_parameters[col_ind], form_parameters[col_ind + 1]):
                tmp = np.hstack([np.zeros(j), m, np.zeros(size - j - 2)])
                A = np.vstack([A, tmp])
    if len(form_parameters) % 2 == 0:
        m = [-1, 1]
        for j in range(form_parameters[-2], form_parameters[-1] - 1):
            tmp = np.hstack([np.zeros(j), m, np.zeros(size - j - 2)])
            A = np.vstack([A, tmp])
    else:
        m = [1, -1]
        for j in range(form_parameters[-2], form_parameters[-1] - 1):
            tmp = np.hstack([np.zeros(j), m, np.zeros(size - j - 2)])
            A = np.vstack([A, tmp])
    return A[1:]


def numeric_sol(f, A):
    C = np.zeros(A.shape[0])
    X = cp.Variable(A.shape[1])
    constraints = [A @ X >= C]
    obj = cp.Minimize(cp.norm(X - f))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return X.value
