# Filename:     Controller.py
# Written by:   Niranjan Bhujel
# Description:  Contains controller such as LQR, MPC, etc.


from math import inf, isinf
from pynlcontrol.BasicUtils import Integrate
import casadi as ca


def LQR(A, B, C, D, Q, R, Qt, Ts, horizon=inf, reftrack=False, NMAX=1000, tol=1e-5, Integrator='rk4'):
    """
    Function to implement discrete-time linear quadratic regulator (LQR).

    Parameters
    ----------
    A : numpy.2darray or casadi.SX.array
        Continuous time state matrix
    B : numpy.2darray or casadi.SX.array
        Continuous time input matrix
    C : numpy.2darray or casadi.SX.array
        Continuous time output matrix
    D : numpy.2darray or casadi.SX.array
        Continuous time output matrix coefficient of input
    Q : numpy.2darray or casadi.SX.array
        Weight to penalize control error
    R : numpy.2darray or casadi.SX.array
        Weight to penalize control effort
    Qt : numpy.2darray or casadi.SX.array
        Weight of terminal cost to penalize control error
    Ts : float
        Sample time of controller
    horizon : int, optional
        Horizon length of LQR. Defaults to inf for infinite horizon LQR problem.
    reftrack : bool, optional
        Whether problem is reference tracking. Defaults to False.
    NMAX : int, optional
        Maximum iteration for solving matrix Ricatti equation. Defaults to 1000.
    tol : float, optional
        Tolerance for solution of matrix Ricatti equation. Defaults to 1e-5.
    Integrator : str, optional
        Integrator to be used for discretization. Defaults to 'rk4'.

    Returns
    -------
    tuple
        Tuple of Input, Output, Input name and Output name. Inputs are x or [x, r] (depending upon the problem is reference tracking or not) and output are u and K.
            Input and output are casadi symbolics (`casadi.SX`).
            These inputs are and outputs can be mapped using `casadi.Function` which can further be code generated.
    
    Example
    -------
    >>> from pynlcontrol import Controller, BasicUtils
    >>> import casadi as ca
    >>> Q11 = ca.SX.sym('Q11')
    >>> Q22 = ca.SX.sym('Q22')
    >>> Q33 = ca.SX.sym('Q33')
    >>> Q = BasicUtils.directSum([Q11, Q22, Q33])
    >>> R11 = ca.SX.sym('R11')
    >>> R22 = ca.SX.sym('R22')
    >>> R = BasicUtils.directSum([R11, R22])
    >>> A = ca.SX([[-0.4,0.1,-2],[0,-0.3,4],[1,0,0]])
    >>> B = ca.SX([[1,1],[0,1],[1,0]])
    >>> C = ca.SX([[1, 0, 0], [0, 1, 0]])
    >>> D = ca.SX([[0, 0], [0, 0]])
    >>> In, Out, InName, OutName = Controller.LQR(A=A, B=B, C=C, D=D, Q=Q, R=R, Qt=Q, Ts=0.1, horizon=10, reftrack=True)
    >>> lqr_func = ca.Function('lqr_func', In + [Q11, Q22, Q33, R11, R22], Out, InName + ['Q11', 'Q22', 'Q33', 'R11', 'R22'], OutName)
    >>> BasicUtils.Gen_Code(lqr_func, 'lqr_code', printhelp=True)
    x(3, 1), ref(2, 1), Q11(1, 1), Q22(1, 1), Q33(1, 1), R11(1, 1), R22(1, 1) -> u(2, 1), K(6, 1)
    lqr_code.c
    lqr_code_Call.c
    #include "lqr_code.h"
    #include "lqr_code_Call.h"
    lqr_code_Call_Func(x, ref, Q11, Q22, Q33, R11, R22, u, K);


    Running above code generates C-codes for LQR implementation. Implementation using Simulink can be found in example folder.
    """

    nX = A.shape[0]
    nU = B.shape[1]
    nY = C.shape[0]

    assert D.shape[1] == B.shape[1], "Inconsistent shape of B and D"
    assert nY == nU, "Number of control inputs and controlled outputs should be same."
    assert nX==Q.shape[0]==Q.shape[1], "Error in size of Q"
    assert nX==Qt.shape[0]==Qt.shape[1], "Error in size of Qt"
    assert nY==R.shape[0]==R.shape[1], "Error in size of R"

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
        r = ca.SX.sym('r', nY, 1)
        tmp = ca.vertcat(
            ca.horzcat(A, B),
            ca.horzcat(C, D)
        )

        Nxu = ca.inv(tmp) @ ca.vertcat(ca.GenSX_zeros(nX, nY), ca.SX_eye(nY))
        Nx = Nxu[0:nX, :]
        Nu = Nxu[nX:,:]

        u += (Nu + K@Nx)@r

        return [x, r], [u, K.T.reshape((-1, 1))], ['x', 'ref'], ['u', 'K']
    else:
        return [x], [u, K.T.reshape((-1, 1))], ['x'], ['u', 'K']
