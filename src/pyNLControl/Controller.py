# Filename:     Controller.py
# Written by:   Niranjan Bhujel
# Description:  Contains controller such as LQR, MPC, etc.


from math import inf, isinf
from pynlcontrol.BasicUtils import Integrate
import casadi as ca


def LQR(A, B, C, D, Q, R, Qt, Ts, horizon=inf, reftrack=False, NMAX=1000, tol=1e-5, Integrator='rk4'):
    """Function to implement discrete-time linear quadratic regulator

    Args:
        A (numpy.2darray or casadi.SX.array): Continuous time state matrix
        B (numpy.2darray or casadi.SX.array): Continuous time input matrix
        C (numpy.2darray or casadi.SX.array): Continuous time output matrix
        D (numpy.2darray or casadi.SX.array): Continuous time output matrix coefficient of input
        Q (numpy.2darray or casadi.SX.array): Weight to penalize control error
        R (numpy.2darray or casadi.SX.array): Weight to penalize control effort
        Qt (numpy.2darray or casadi.SX.array): Weight of terminal cost to penalize control effort
        Ts (float): Sample time of controller
        horizon (int, optional): Horizon length of LQR. Defaults to inf for infinite horizon LQR problem
        reftrack (bool, optional): Whether problem is reference tracking. Defaults to False.
        NMAX (int, optional): Maximum iteration for solving matrix Ricatti equation. Defaults to 1000.
        tol (float, optional): Tolerance for solution of matrix Ricatti equation. Defaults to 1e-5.
        Integrator (str, optional): Integrator to be used for discretization. Defaults to 'rk4'.

    Returns:
        tuple: Tuple of Input, Output, Input name and Output name. Inputs are x or [x, r] 
            (depending upon the problem is reference tracking or not) and output are u and K.
            Input and output are casadi symbolics (`casadi.SX`).

            These inputs are and outputs can be mapped using `casadi.Function` which can further be code generated.
    """

    nX = A.shape[0]
    nU = B.shape[1]
    x = ca.SX.sym('x', nX, 1)
    u = ca.SX.sym('u', nU, 1)

    def Fc(x, u):
        return A @ x + B @ u

    xk1 = Integrate(Fc, Integrator, Ts, x, u)
    Ad = ca.jacobian(xk1, x)
    Bd = ca.jacobian(xk1, u)

    P = Qt
    MAXITER = NMAX if isinf(horizon) else horizon
    for _ in range(MAXITER):
        P = Ad.T @ P @ Ad - (Ad.T @ P @ Bd) @ ca.inv(R +
                                                     Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad) + Q

    K = ca.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

    u = -K @ x

    if reftrack:
        r = ca.SX.sym('r', C.shape[0], 1)
        tmp = ca.vertcat(
            ca.horzcat(A, B),
            ca.horzcat(C, D)
        )

        Nxu = ca.inv(tmp) @ ca.vertcat(ca.GenSX_zeros(nX, 1), ca.GenSX_ones(nU, 1))
        Nx = Nxu[0:nX, :]
        Nu = Nxu[nX:,:]

        u += (Nu + K@Nx)@r

        return [x, r], [u, K], ['x', 'ref'], ['u', 'K']
    else:
        return [x], [u, K], ['x'], ['u', 'K']
