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

    def phi_update(self, coherent_state):

        [P, aIBi, mI, mB, n0, gBB] = self.Params

        dv = coherent_state.dV

        amplitude_t = coherent_state.amplitude
        PB_t = coherent_state.get_PhononMomentum()

        betaSum = amplitude_t + np.conjugate(amplitude_t)
        # print(betaSum.shape)
        # print(dv.shape)
        # print(self.Wk_grid.shape)
        xp_t = 0.5 * np.dot(self.Wk_grid, betaSum * dv)

        return self.gnum * n0 + self.gnum * np.sqrt(n0) * xp_t + (P**2 - PB_t**2) / (2 * mI)

    def amplitude_update(self, coherent_state):
        # here on can write any method induding Runge-Kutta 4

        [P, aIBi, mI, mB, n0, gBB] = self.Params

        dv = coherent_state.dV

        amplitude_t = coherent_state.amplitude
        PB_t = coherent_state.get_PhononMomentum()

        betaSum = amplitude_t + np.conjugate(amplitude_t)
        xp_t = 0.5 * np.dot(self.Wk_grid, betaSum * dv)

        betaDiff = amplitude_t - np.conjugate(amplitude_t)
        xm_t = 0.5 * np.dot(self.Wki_grid, betaDiff * dv)

        return -1j * (self.gnum * np.sqrt(n0) * self.Wk_grid +
                      amplitude_t * (self.Omega0_grid - self.kcos * (P - PB_t) / mI) +
                      self.gnum * (self.Wk_grid * xp_t + self.Wki_grid * xm_t))
