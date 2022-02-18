# Filename:     Controller.py
# Written by:   Niranjan Bhujel
# Description:  Contains controller such as LQR, MPC, etc.


from math import inf, isinf
from pynlcontrol.BasicUtils import Integrate, nlp2GGN, casadi2List, directSum, __SXname__
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
    assert nX == Q.shape[0] == Q.shape[1], "Error in size of Q"
    assert nX == Qt.shape[0] == Qt.shape[1], "Error in size of Qt"
    assert nY == R.shape[0] == R.shape[1], "Error in size of R"

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
        Nu = Nxu[nX:, :]

        u += (Nu + K@Nx)@r

        return [x, r], [u, K.T.reshape((-1, 1))], ['x', 'ref'], ['u', 'K']
    else:
        return [x], [u, K.T.reshape((-1, 1))], ['x'], ['u', 'K']


def simpleMPC(nX, nU, nY, Fc, Hc, N, Ts, uLow, uUpp, GGN=False, Integrator='rk4', Options=None):
    """
    Function to generate simple MPC code using `qrqp` solver. For use with other advanced solver, see `MPC` class.

    Parameters
    ----------
    nX : int
        Number of state variables.
    nU : int
        Number of input variables
    nY : int
        Number of control output variables
    Fc : function
        Function that returns right hand side of state equation.
    Hc : function
        Function that returns right hand side of control output equation.
    N : float or casadi.SX array or numpy.2darray
        Horizon length
    Ts : float
        Sample time
    uLow : list or float
        Lower limit on control input
    uUpp : list of str
        Upper limit on control input
    GGN : bool, optional
        Whether generalized Gauss Newton should be used. Use only for nonlinear problem. by default False
    Integrator : str, optional
        Integration method. See `BasicUtils.Integrate()` function. by default 'rk4'
    Options : _type_, optional
        Option for `qrqp` solver. Defaults to None.

    Returns
    -------
    tuple:
        Tuple of Input, Output, Input name and Output name. Input and output are list of casadi symbolics (`casadi.SX`).
            Inputs are initial guess, current state, reference, corresponding weights
            Outputs value of all decision variables, calculated control signal and cost function

    Example
    -------
    >>> import casadi as ca
    >>> from pynlcontrol import BasicUtils, Controller
    >>> def Fc(x, u):    
        A = ca.SX([[-0.4, 0.1, -2], [0, -0.3, 4], [1, 0, 0]])
        B = ca.SX([[1, 1], [0, 1], [1, 0]])
        return A @ x + B @ u
    >>> def Hc(x):
        return ca.vertcat(x[0], x[1])
    >>> In, Out, InName, OutName = Controller.simpleMPC(3, 2, 2, Fc, Hc, 25, 0.1, [-10, 0], [10, 3], GGN=False)
    -------------------------------------------
    This is casadi::QRQP
    Number of variables:                             128
    Number of constraints:                            78
    Number of nonzeros in H:                         100
    Number of nonzeros in A:                         453
    Number of nonzeros in KKT:                      1112
    Number of nonzeros in QR(R):                    1728
    -------------------------------------------
    This is casadi::Sqpmethod.
    Using exact Hessian
    Number of variables:                             128
    Number of constraints:                            78
    Number of nonzeros in constraint Jacobian:       453
    Number of nonzeros in Lagrangian Hessian:        100

    >>> MPC_func = ca.Function('MPC_func', In, Out, InName, OutName)
    >>> BasicUtils.Gen_Code(MPC_func, 'MPC_Code', printhelp=True, optim=True)
    zGuess(128, 1), x0(3, 1), xref(2, 1), Qp11(1, 1), Qp22(1, 1), Qtp11(1, 1), Qtp22(1, 1), Rp11(1, 1), Rp22(1, 1) -> zOut(128, 1), uCalc(2, 1), Cost(1, 1)
    MPC_Code.c
    MPC_Code_Call.c
    #include "MPC_Code.h"
    #include "MPC_Code_Call.h"
    MPC_Code_Call_Func(zGuess, x0, xref, Qp11, Qp22, Qtp11, Qtp22, Rp11, Rp22, zOut, uCalc, Cost);
    """
    X = ca.SX.sym('X', nX, N+1)
    U = ca.SX.sym('U', nU, N)
    X0 = ca.SX.sym('X0', nX, 1)
    Xref = ca.SX.sym('Xref', nY, 1)

    Q = [ca.SX.sym(f'Q{k+1}{k+1}') for k in range(nY)]
    Qt = [ca.SX.sym(f'Qt{k+1}{k+1}') for k in range(nY)]
    R = [ca.SX.sym(f'R{k+1}{k+1}') for k in range(nU)]

    Qw = directSum(Q)
    Qtw = directSum(Qt)
    Rw = directSum(R)

    J = ca.vertcat()
    for k in range(N):
        J = ca.vertcat(
            J,
            Qw @ (Hc(X[:,k]) - Xref),
            Rw @ U[:,k]
        )

    J = ca.vertcat(
        J,
        Qtw @ (Hc(X[:,k]) - Xref)
    )

    g = ca.vertcat()
    lbg = ca.vertcat()
    ubg = ca.vertcat()
    # Initial state constraints
    g = ca.vertcat(
        g,
        X[:, 0] - X0,
    )
    lbg = ca.vertcat(
        lbg,
        ca.GenSX_zeros(nX, 1),
    )
    ubg = ca.vertcat(
        ubg,
        ca.GenSX_zeros(nX, 1),
    )

    # State equations constraints
    for k in range(N):
        g = ca.vertcat(
            g,
            X[:, k+1] - Integrate(Fc, Integrator, Ts, X[:, k], U[:, k]),
        )
        lbg = ca.vertcat(
            lbg,
            ca.GenSX_zeros(nX, 1),
        )
        ubg = ca.vertcat(
            ubg,
            ca.GenSX_zeros(nX, 1),
        )

    lbx = ca.vertcat()
    ubx = ca.vertcat()
    for k in range(N):
        lbx = ca.vertcat(
            lbx,
            ca.vertcat(*uLow)
        )
        ubx = ca.vertcat(
            ubx,
            ca.vertcat(*uUpp)
        )

    for k in range(N+1):
        lbx = ca.vertcat(
            lbx,
            -ca.inf*ca.GenSX_ones(nX, 1)
        )
        ubx = ca.vertcat(
            ubx,
            ca.inf*ca.GenSX_ones(nX, 1)
        )

    z = ca.vertcat(
        U.reshape((-1, 1)),
        X.reshape((-1, 1))
    )
    pIn = ca.vertcat(X0, Xref, *Q, *Qt, *R)
    if GGN:
        nlp = nlp2GGN(z, J, g, lbg, ubg, pIn)
        nlp['p'] = ca.vertcat(
            nlp['p'],
            nlp['zOp'],
        )
    else:
        nlp = {
            'x': z,
            'f': ca.norm_2(J)**2,
            'g': g,
            'lbg': lbg,
            'ubg': ubg,
            'p': pIn
        }

    optTemp = {'qpsol': 'qrqp'}
    if Options is not None:
        Options.update(optTemp)
    else:
        Options = optTemp

    MPC_prob = {
        'x': nlp['x'],
        'f': nlp['f'],
        'g': nlp['g'],
        'p': nlp['p']
    }

    S = ca.nlpsol('S', 'sqpmethod', MPC_prob, Options)

    zGuess = ca.MX.sym('zGuess', MPC_prob['x'].shape)
    X0p = ca.MX.sym('X0p', nX, 1)

    Xrefp = ca.MX.sym('Xrefp', nY, 1)

    Qp = [ca.MX.sym(f'Qp{k+1}{k+1}') for k in range(nY)]
    Qtp = [ca.MX.sym(f'Qtp{k+1}{k+1}') for k in range(nY)]
    Rp = [ca.MX.sym(f'Rp{k+1}{k+1}') for k in range(nU)]

    pVal = pIn = ca.vertcat(X0p, Xrefp, *Qp, *Qtp, *Rp)

    if GGN:
        zOpp = ca.MX.sym('zOpp', z.shape)
        pVal = ca.vertcat(
            pVal,
            zOpp
        )

    r = S(
        x0=zGuess,
        p=pVal,
        lbg=casadi2List(lbg),
        ubg=casadi2List(ubg),
        lbx=casadi2List(lbx),
        ubx=casadi2List(ubx),
    )

    In = [
        zGuess,
        X0p,
        Xrefp] + Qp + Qtp + Rp
    InName = [
        'zGuess',
        'x0',
        'xref'] + __SXname__(Qp) + __SXname__(Qtp) + __SXname__(Rp)

    if GGN:
        In.append(zOpp)
        InName.append('zOp')

    Out = [r['x']]
    OutName = ['zOut']

    Out.append(r['x'][0:nU])
    OutName.append('uCalc')

    Out.append(r['f'])
    OutName.append('Cost')

    return In, Out, InName, OutName
