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


def phase_update(variables_t, t, coherent_state, hamiltonian):

    [P, aIBi, mI, mB, n0, gBB] = hamiltonian.Params

    # Here I need the original grid
    dv = coherent_state.grid.dV()

    # Split variables into x and p
    [x_t, p_t] = np.split(coherent_state.amplitude, 2)
    PB_t = coherent_state.get_PhononMomentum()

    return hamiltonian.gnum * n0 + hamiltonian.gnum * np.sqrt(n0) * np.dot(hamiltonian.Wk_grid, x_t * dv) +\
        (P**2 - PB_t**2) / (2 * mI)


class CoherentState:
    # """ This is a class that stores information about coherent state """

    def __init__(self, grid_space):

        size = grid_space.size()
        self.amplitude = np.zeros(2 * size, dtype=float)
        self.phase = 0
        self.grid = grid_space

        self.dV = np.append(grid_space.dV(), grid_space.dV())
        self.kcos = np.append(kcos_func(self.grid), kcos_func(self.grid))
        # self.sigma = np.bmat([[np.zeros(size, size), np.identity(size)],
        #                      [np.zeros(size, size), np.identity(size)]])
    # EVOLUTION

    def evolve(self, dt, hamiltonian):

        # ODE solver parameters: absolute and relevant error
        abserr = 1.0e-8
        relerr = 1.0e-6

        # Create the time samples for the output of the ODE solver.
        t = [0, dt]

        # Call the ODE solver.
        amplitude_sol = odeint(amplitude_update, self.amplitude, t, args=(self, hamiltonian),
                               atol=abserr, rtol=relerr)
        phase_sol = odeint(phase_update, self.phase, t, args=(self, hamiltonian),
                           atol=abserr, rtol=relerr)

        # Overrite the solution to its container
        self.amplitude = amplitude_sol[-1]
        self.phase = phase_sol[-1]

    # OBSERVABLES

    def get_PhononNumber(self):

        coherent_amplitude = self.amplitude
        return 0.5 * np.dot(coherent_amplitude * coherent_amplitude, self.dV)

    def get_PhononMomentum(self):

        coherent_amplitude = self.amplitude
        return np.dot(self.kcos, coherent_amplitude * coherent_amplitude * self.dV)

    def get_DynOverlap(self):
        # dynamical overlap/Ramsey interferometry signal
        NB_vec = self.get_PhononNumber()
        exparg = -1j * self.phase - (1 / 2) * NB_vec
        return np.exp(exparg)
