import numpy as np
from polaron_functions import kcos_func
from scipy.integrate import odeint
from PolaronHamiltonian import var_update
from copy import copy


class CoherentState:
    # """ This is a class that stores information about coherent state """

    def __init__(self, grid_space):

        self.size = grid_space.size()
        self.variables = np.zeros(2 * self.size + 1, dtype=float)
        self.grid = grid_space

        self.dV = grid_space.dV()

    # EVOLUTION

    def evolve(self, dt, hamiltonian):

        # ODE solver parameters: absolute and relevant error
        abserr = 1.0e-10
        relerr = 1.0e-10

        # Create the time samples for the output of the ODE solver.
        t = [0, dt]

        # Call the ODE solver.

        var_sol = odeint(var_update, self.variables, t, args=(self, hamiltonian),
                         atol=abserr, rtol=relerr)

        # Overrite the solution to its container
        self.variables = var_sol[-1]

    # OBSERVABLES

    def get_PhononNumber(self, hamiltonian):

        x_t = self.variables[0:self.size]
        p_t = self.variables[self.size: (2 * self.size)]

        return 0.5 * np.dot((x_t**2 + p_t**2), self.dV)

    def get_PhononMomentum(self, hamiltonian):

        x_t = self.variables[0:self.size]
        p_t = self.variables[self.size: (2 * self.size)]

        return 0.5 * np.dot(hamiltonian.kcos, (x_t**2 + p_t**2) * self.dV)

    def get_DynOverlap(self, hamiltonian):
        # dynamical overlap/Ramsey interferometry signal
        NB_t = self.get_PhononNumber(hamiltonian)
        exparg = -1j * self.variables[-1] - (1 / 2) * NB_t

        return np.exp(exparg)

    def get_MomentumDispersion(self, hamiltonian):

        x_t = self.variables[0:self.size]
        p_t = self.variables[self.size: (2 * self.size)]

        return 0.5 * np.dot(hamiltonian.kpow2, (x_t**2 + p_t**2) * self.dV)
