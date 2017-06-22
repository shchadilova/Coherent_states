import numpy as np
import polaron_functions as pf


class PolaronHamiltonian:
        # """ This is a class that stores information about the Hamiltonian"""

    def __init__(self, coherent_state, Params):

        # Params = [P, aIBi, mI, mB, n0, gBB]
        self.Params = Params

        self.gnum = pf.g(coherent_state.grid, *Params)
        self.Omega0_grid = pf.omega0(coherent_state.grid, *Params)
        self.Wk_grid = pf.Wk(coherent_state.grid, *Params)
        self.Wki_grid = 1 / self.Wk_grid
        self.kcos = pf.kcos_func(coherent_state.grid)
        self.kpow2 = pf.kpow2_func(coherent_state.grid)
        self.dv = coherent_state.dV

        size = coherent_state.size
        self.size = size
        # self.h_linear = np.zeros(2 * size + 1, dtype=float)


def var_update(variables_t, t, coherent_state, hamiltonian):
    # here on can write any method induding Runge-Kutta 4

    [P, aIBi, mI, mB, n0, gBB] = hamiltonian.Params
    # Here I need an original grid
    dv = coherent_state.dV
    size = coherent_state.size

    equtions = np.zeros(2 * size + 1, dtype=float)

    # Split variables into x and p
    x_t = variables_t[0:size]
    p_t = variables_t[size: (2 * size)]

    PB_t = 0.5 * np.dot(hamiltonian.kcos, (x_t**2 + p_t**2) * dv)

    # equation for x_t
    equtions[0:size] = (p_t * (hamiltonian.Omega0_grid - hamiltonian.kcos * (P - PB_t) / mI) +
                        (hamiltonian.gnum * np.dot(hamiltonian.Wki_grid, p_t * dv)) * hamiltonian.Wki_grid)

    # equation for p_t
    equtions[size:2 * size] = (-1) * (2 * hamiltonian.gnum * np.sqrt(n0) * hamiltonian.Wk_grid +
                                      x_t * (hamiltonian.Omega0_grid - hamiltonian.kcos * (P - PB_t) / mI) +
                                      hamiltonian.gnum * hamiltonian.Wk_grid * np.dot(hamiltonian.Wk_grid, x_t * dv))
    # equation for phi
    equtions[-1] = (hamiltonian.gnum * n0 + hamiltonian.gnum * np.sqrt(n0) * np.dot(hamiltonian.Wk_grid, x_t * dv) +
                    (P**2 - PB_t**2) / (2 * mI))

    return equtions
