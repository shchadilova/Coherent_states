import numpy as np
from polaron_functions import kcos_func
from scipy.integrate import odeint


def amplitude_update(variables_t, t, coherent_state, hamiltonian):
    # here on can write any method induding Runge-Kutta 4

    [P, aIBi, mI, mB, n0, gBB] = hamiltonian.Params

    # Here I need an original grid
    dv = coherent_state.grid.dV()

    # Split variables into x and p
    [x_t, p_t] = np.split(variables_t, 2)
    PB_t = coherent_state.get_PhononMomentum()

    h_x = 2 * hamiltonian.gnum * np.sqrt(n0) * hamiltonian.Wk_grid +\
        x_t * (hamiltonian.Omega0_grid - hamiltonian.kcos * (P - PB_t) / mI) +\
        hamiltonian.gnum * hamiltonian.Wk_grid * np.dot(hamiltonian.Wk_grid, x_t * dv)
    h_y = p_t * (hamiltonian.Omega0_grid - hamiltonian.kcos * (P - PB_t) / mI) +\
        hamiltonian.gnum * hamiltonian.Wki_grid * np.dot(hamiltonian.Wki_grid, p_t * dv)

    return np.append(h_y, -1 * h_x)


class CoherentState:
    # """ This is a class that stores information about coherent state """

    def __init__(self, grid_space):

        size = grid_space.size()
        self.amplitude = np.zeros(2 * size, dtype=float)
        self.phase = 0 + 0j
        self.grid = grid_space

        self.dV = np.append(grid_space.dV(), grid_space.dV())
        self.kcos = np.append(kcos_func(self.grid), kcos_func(self.grid))
        # self.sigma = np.bmat([[np.zeros(size, size), np.identity(size)],
        #                      [np.zeros(size, size), np.identity(size)]])
    # EVOLUTION

    def evolve(self, dt, hamiltonian):

        # self.phase = self.phase + dt * hamiltonian.phi_update(self)
        # self.amplitude = self.amplitude + dt * hamiltonian.amplitude_update(self)
        # ODE solver parameters
        # ODE solver parameters
        abserr = 1.0e-8
        relerr = 1.0e-6
        stoptime = dt
        numpoints = 2

        # Create the time samples for the output of the ODE solver.
        # I use a large number of points, only because I want to make
        # a plot of the solution that looks nice.
        t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

        # Call the ODE solver.
        wsol = odeint(amplitude_update, self.amplitude, t, args=(self, hamiltonian),
                      atol=abserr, rtol=relerr)

        self.amplitude = wsol[-1]

    # OBSERVABLES

    def get_PhononNumber(self):

        coherent_amplitude = self.amplitude
        return 0.5 * np.dot(coherent_amplitude * coherent_amplitude, self.dV)

    def get_PhononMomentum(self):

        coherent_amplitude = self.amplitude
        return np.dot(self.kcos, coherent_amplitude * coherent_amplitude * self.dV)

    # def get_DynOverlap(self):
    #     # dynamical overlap/Ramsey interferometry signal
    #     NB_vec = self.get_PhononNumber()
    #     exparg = -1j * self.phase - (1 / 2) * NB_vec
    #     return np.exp(exparg)
