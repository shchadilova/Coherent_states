import numpy as np
import polaron_functions as pf


class PolaronHamiltonian:
        # """ This is a class that stores information about the Hamiltonian"""

    def __init__(self, coherent_state, Params):

        # Params = [P, aIBi, mI, mB, n0, gBB]
        self.Params = Params

        self.grid = coherent_state.grid

        self.gnum = pf.g(self.grid, *Params)
        self.Omega0_grid = pf.omega0(self.grid, *Params)
        self.Wk_grid = pf.Wk(self.grid, *Params)
        self.Wki_grid = 1 / self.Wk_grid
        self.kcos = pf.kcos_func(self.grid)

        # print(self.Omega0_grid.shape)

    # def phi_update(self, coherent_state):

    #     [P, aIBi, mI, mB, n0, gBB] = self.Params

    #     # Here I need the original grid
    #     dv = coherent_state.grid.dV()

    #     # Split variables into x and p
    #     [x_t, p_t] = np.split(coherent_state.amplitude, 2)
    #     PB_t = coherent_state.get_PhononMomentum()

    #     return self.gnum * n0 + self.gnum * np.sqrt(n0) * np.dot(self.Wk_grid, x_t * dv) +\
    #         (P**2 - PB_t**2) / (2 * mI)

    # def amplitude_update(self, coherent_state):
    #     # here on can write any method induding Runge-Kutta 4

    #     [P, aIBi, mI, mB, n0, gBB] = self.Params

    #     # Here I need the original grid
    #     dv = coherent_state.grid.dV()

    #     # Split variables into x and p
    #     [x_t, p_t] = np.split(coherent_state.amplitude, 2)
    #     PB_t = coherent_state.get_PhononMomentum()

    #     h_x = 2 * self.gnum * np.sqrt(n0) * self.Wk_grid +\
    #         x_t * (self.Omega0_grid - self.kcos * (P - PB_t) / mI) +\
    #         self.gnum * self.Wk_grid * np.dot(self.Wk_grid, x_t * dv)
    #     h_y = p_t * (self.Omega0_grid - self.kcos * (P - PB_t) / mI) +\
    #         self.gnum * self.Wki_grid * np.dot(self.Wki_grid, p_t * dv)

    #     return np.append(h_y, -1 * h_x)
