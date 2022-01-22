from pyNLControl import BasicUtils
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square

# Continuous time ODE
def Fc(x, u, R, L, C):
    iL = x[0]
    vC = x[1]

    vi = u[0]

    return np.array(
        [(vi - vC)/L,
         (iL - vC/R)/C]
    )


# Parameter values
R = 10
L = 0.1
C = 1000e-6

# Sampling time, frequency of input
Ts = 1e-3
f = 5


t = np.arange(0, 1, Ts)

x0 = np.array([0, 0])
x = np.zeros((2, t.shape[0]))
x[:, 0] = x0

vi = square(2 * np.pi * f * t, 0.5).reshape((1, -1))
for k in range(0, t.shape[0]-1):
    x[:, k+1] = BasicUtils.Integrate(Fc, 'rk4', Ts, x[:, k], vi[:, k], R, L, C)

# Plot the results
fig, ax = plt.subplots(3, 1, figsize=(3.3, 6), constrained_layout=True)

ax[0].plot(t, vi[0,:])
ax[0].set_xlabel('time (s)')
ax[0].set_ylabel('$v_i$ (V)')

ax[1].plot(t, x[0,:])
ax[1].set_xlabel('time (s)')
ax[1].set_ylabel('$i_L$ (A)')

ax[2].plot(t, x[1,:])
ax[2].set_xlabel('time (s)')
ax[2].set_ylabel('$v_C$ (V)')

plt.show()
