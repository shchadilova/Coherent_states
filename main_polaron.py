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

# cProfile.run('re.compile("foo|bar")')
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
P = 1.
aIBi = -1.

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
MomDisp_Vec = np.zeros(tVec.size, dtype=float)

for ind, t in enumerate(tVec):

    PB_Vec[ind] = cs.get_PhononMomentum(ham)
    NB_Vec[ind] = cs.get_PhononNumber(ham)
    DynOv_Vec[ind] = cs.get_DynOverlap(ham)
    MomDisp_Vec[ind] = cs.get_MomentumDispersion(ham)
    cs.evolve(dt, ham)

end = timer()

print(end - start)

figN, axN = plt.subplots()
axN.plot(tVec, MomDisp_Vec, 'k-')
axN.set_xlabel('Time ($t$)')
axN.set_ylabel('$\Delta P^2_{imp}$')
axN.set_title('Number of Phonons')
figN.savefig('quaench_momdisp.pdf')

# figN, axN = plt.subplots()
# axN.plot(tVec, NB_Vec, 'k-')
# axN.set_xlabel('Time ($t$)')
# axN.set_ylabel('$N_{ph}$')
# axN.set_title('Number of Phonons')
# figN.savefig('quaench_momdisp.pdf')

plt.show()
