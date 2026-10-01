import numpy as np

from src.vessel import Vessel
from src.plotting import plot_vessel_motion

vessel = Vessel()

dt = 0.1
simulation_time = 100.0
N = int(simulation_time/dt)

tau = np.array([
    50_000.0,
    50000.0,
    0.0
])

states = np.zeros((N, 6))

for k in range(N):

    state = vessel.step(tau, dt)

    states[k] = state

states = np.array(states)

plot_vessel_motion(states)