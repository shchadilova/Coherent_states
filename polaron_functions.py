import numpy as np

# ---- BASIS FUNCTIONS ----


def ur(mI, mB):
    return (mB * mI) / (mB + mI)


def eB(k, mB):
    return k**2 / (2 * mB)


def w(k, gBB, mB, n0):
    return np.sqrt(eB(k, mB) * (eB(k, mB) + 2 * gBB * n0))


# ---- COMPOSITE FUNCTIONS ----


def g(grid_space, P, aIBi, mI, mB, n0, gBB):
    # gives bare interaction strength constant
    k_max = grid_space.arrays['k'][-1]
    mR = ur(mI, mB)
    return 1 / ((mR / (2 * np.pi)) * aIBi - (mR / np.pi**2) * k_max)


def omega0(grid_space, P, aIBi, mI, mB, n0, gBB):
    #
    names = list(grid_space.arrays.keys())
    functions_omega0 = [lambda k: w(k, gBB, mB, n0) + (k**2 / (2 * mI)), lambda th: 0 * th + 1]
    return grid_space.function_prod(names, functions_omega0)


def Wk(grid_space, P, aIBi, mI, mB, n0, gBB):
    #
    names = list(grid_space.arrays.keys())
    functions_wk = [lambda k: np.sqrt(eB(k, mB) / w(k, gBB, mB, n0)), lambda th: 0 * th + 1]
    return grid_space.function_prod(names, functions_wk)


def kcos_func(grid_space):
    #
    names = list(grid_space.arrays.keys())
    functions_kcos = [lambda k: k, np.cos]
    return grid_space.function_prod(names, functions_kcos)
