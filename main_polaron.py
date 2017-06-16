from timeit import default_timer as timer
import numpy as np
import os
import Grid
import CoherentState
import PolaronHamiltonian
import matplotlib
import matplotlib.pyplot as plt
import cProfile
import re

cProfile.run('re.compile("foo|bar")')
matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})


# Initialization Grid
k_max = 10
dk = 0.05
Ntheta = 10
dtheta = np.pi / (Ntheta - 1)

grid_space = Grid.Grid("SPHERICAL_2D")
grid_space.init1d('k', dk, k_max, dk)
grid_space.init1d('th', dtheta, np.pi, dtheta)

# Initialization CoherentState
cs = CoherentState.CoherentState(grid_space)

# Initialization PolaronHamiltonian

mI = 1
mB = 1
n0 = 1
gBB = (4 * np.pi / mB) * 0.05
P = 0.1
aIBi = -10

Params = [P, aIBi, mI, mB, n0, gBB]
ham = PolaronHamiltonian.PolaronHamiltonian(cs, Params)


# Time evolution
tMax = 10
dt = 0.1

start = timer()

tVec = np.arange(0, tMax, dt)
PB_Vec = np.zeros(tVec.size, dtype=float)
NB_Vec = np.zeros(tVec.size, dtype=float)
DynOv_Vec = np.zeros(tVec.size, dtype=complex)

for ind, t in enumerate(tVec):
    PB_Vec[ind] = cs.get_PhononMomentum()
    NB_Vec[ind] = cs.get_PhononNumber()
    DynOv_Vec[ind] = cs.get_DynOverlap()

    cs.evolve(dt, ham)

end = timer()

print(end - start)

figN, axN = plt.subplots()
axN.plot(tVec, NB_Vec, 'k-')
axN.set_xlabel('Time ($t$)')
axN.set_ylabel('$N_{ph}$')
axN.set_title('Number of Phonons')
figN.savefig('quench_PhononNumber.pdf')

plt.show()
