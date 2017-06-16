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
        self.dv = coherent_state.grid.dV()


def amplitude_update(variables_t, t, coherent_state, hamiltonian):
    # here on can write any method induding Runge-Kutta 4

    [P, aIBi, mI, mB, n0, gBB] = hamiltonian.Params

    # Here I need an original grid
    dv = coherent_state.grid.dV()

    # Split variables into x and p
    [x_t, p_t] = np.split(variables_t, 2)
    PB_t = 0.5 * (variables_t * variables_t) @ coherent_state.dV

    h_x = 2 * hamiltonian.gnum * np.sqrt(n0) * hamiltonian.Wk_grid +\
        x_t * (hamiltonian.Omega0_grid - hamiltonian.kcos * (P - PB_t) / mI) +\
        hamiltonian.gnum * hamiltonian.Wk_grid * np.dot(hamiltonian.Wk_grid, x_t * dv)
    h_y = p_t * (hamiltonian.Omega0_grid - hamiltonian.kcos * (P - PB_t) / mI) +\
        hamiltonian.gnum * hamiltonian.Wki_grid * np.dot(hamiltonian.Wki_grid, p_t * dv)

    return np.append(h_y, -1 * h_x)


def phase_update(variables_t, t, coherent_state, hamiltonian):

    [P, aIBi, mI, mB, n0, gBB] = hamiltonian.Params

    # Here I need the original grid
    dv = coherent_state.grid.dV()

    # Split variables into x and p
    [x_t, p_t] = np.split(coherent_state.amplitude, 2)
    PB_t = 0.5 * np.dot(coherent_state.amplitude * coherent_state.amplitude, coherent_state.dV)

    return hamiltonian.gnum * n0 + hamiltonian.gnum * np.sqrt(n0) * np.dot(hamiltonian.Wk_grid, x_t * dv) +\
        (P**2 - PB_t**2) / (2 * mI)
