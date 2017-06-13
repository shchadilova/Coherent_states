import numpy as np
from polaron_functions import kcos_func


class CoherentState:
    # """ This is a class that stores information about coherent state """

    def __init__(self, grid_space):

        size = grid_space.size()
        self.amplitude = np.zeros(size, dtype=complex)
        self.phase = 0 + 0j
        self.grid = grid_space

        self.dV = grid_space.dV()
        self.kcos = kcos_func(self.grid)
    # EVOLUTION

    def evolve(self, dt, hamiltonian):

        self.phase = self.phase + dt * hamiltonian.phi_update(self)
        self.amplitude = self.amplitude + dt * hamiltonian.amplitude_update(self)

    # OBSERVABLES
    def get_PhononNumber(self):

        coherent_amplitude = self.amplitude
        return np.dot(coherent_amplitude * np.conjugate(coherent_amplitude), self.dV)

    def get_PhononMomentum(self):

        coherent_amplitude = self.amplitude
        return np.dot(self.kcos, coherent_amplitude * np.conjugate(coherent_amplitude) * self.dV)

    def get_DynOverlap(self):
        # dynamical overlap/Ramsey interferometry signal
        NB_vec = self.get_PhononNumber()
        exparg = -1j * self.phase - (1 / 2) * NB_vec
        return np.exp(exparg)
