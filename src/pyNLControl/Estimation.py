from pyNLControl.BasicUtils import Integrate
import casadi as ca


def KF(nX, nU, nY, Ad, Bd, Cd, Qw, Rv):
    """Function to implement Kalman filter. 

    Args:
        nX (int): Number of state variables
        nU (int): Number of control inputs
        nY (int): Number of measurement outputs
        Ad (numpy.2darray or casadi.SX array): Discrete-time state matrix of the system
        Bd (numpy.2darray or casadi.SX array): Discrete-time input matrix of the system
        Cd (numpy.2darray or casadi.SX array): Discrete-time measurement matrix of the system
        Qw (numpy.2darray or casadi.SX array): Process noise covariance matrix
        Rv (numpy.2darray or casadi.SX array): Measurement noise covariance matrix

    Returns:
        tuple: Tuple of Input, Output, Input name and Output name. Inputs are u, y, xp, Pp and output are xhat and Phat. Input and output are casadi symbolics (`casadi.SX`).
            u: Current input to the system
            y: Current measurement of the system
            xp: State estimate from previous discrete time
            Pp: Covariance estimate from previous discrete time (reshaped to column matrix)
            xhat: State estimate at current discrete time
            Phat: Covariance estimate at current discrete time (reshaped to column matrix)

            These inputs are and outputs can be mapped using `casadi.Function` which can further be code generated.
    """
    xp = ca.SX.sym('xp', nX, 1)
    u = ca.SX.sym('u', nU, 1)
    y = ca.SX.sym('y', nY, 1)

    Pp = ca.SX.sym('Pp', nX*nX, 1)
    Ppt = ca.reshape(Pp, nX, nX)

    xkm = Ad @ xp + Bd @ u
    Pkm = Ad @ Ppt @ ca.transpose(Ad) + Qw
    yr = y - Cd @ xkm
    Sk = Cd @ Pkm @ ca.transpose(Cd) + Rv

    Kk = Pkm @ ca.transpose(Cd) @ ca.inv(Sk)

    xhat = xkm + Kk @ yr
    Phat = (ca.SX_eye(Ad.shape[0]) - Kk @ Cd) @ Pkm

    return [u, y, xp, Pp], [xhat, ca.reshape(Phat, nX*nX, 1)], ['u', 'y', 'xhatp', 'Pkp'], ['xhat', 'Phat']


def EKF(nX, nU, ny, F, H, Qw, Rv, Ts, Integrator='rk4'):
    """Function to implement Extended Kalman filter.

    Args:
        nX (int): Number of state variables
        nU (int): Number of control inputs
        ny (int): Number of measurement outputs
        F (function): Function that returns right-hand side of state differential equation
        H (function): Function that retuns measurement variable from state variable
        Qw (numpy.2darray or casadi.SX array): Process noise covariance matrix
        Rv (numpy.2darray or casadi.SX array): Measurement noise covariance matrix
        Ts (float): Sample time of the Kalman filter.
        Integrator (str, optional): Integration method. Defaults to 'rk4'. For list of supported integrator, please see documentation of function `Integrate`.

    Returns:
        tuple: Tuple of Input, Output, Input name and Output name. Inputs are u, y, xp, Pp and output are xhat and Phat. Input and output are casadi symbolics (`casadi.SX`).
            u: Current input to the system
            y: Current measurement of the system
            xp: State estimate from previous discrete time
            Pp: Covariance estimate from previous discrete time (reshaped to column matrix)
            xhat: State estimate at current discrete time
            Phat: Covariance estimate at current discrete time (reshaped to column matrix)

            These inputs are and outputs can be mapped using `casadi.Function` which can further be code generated.
    """
    xp = ca.SX.sym('xp', nX, 1)
    u = ca.SX.sym('u', nU, 1)
    y = ca.SX.sym('y', ny, 1)

    Pp = ca.SX.sym('Pp', nX*nX, 1)
    Ppt = ca.reshape(Pp, nX, nX)

    xkm = Integrate(F, Integrator, Ts, xp, u)
    Fk = ca.jacobian(xkm, xp)

    Pkm = Fk @ Ppt @ ca.transpose(Fk) + Qw
    yr = y - H(xkm)

    tmpfun = ca.Function('tmpfun', [xp], [ca.jacobian(H(xp), xp)])
    Hk = tmpfun(xkm)

    Sk = Hk @ Pkm @ ca.transpose(Hk) + Rv
    Kk = Pkm @ ca.transpose(Hk) @ ca.inv(Sk)

    xhat = xkm + Kk @ yr
    Phat = (ca.SX_eye(Fk.shape[0]) - Kk @ Hk) @ Pkm

    return [u, y, xp, Pp], [xhat, ca.reshape(Phat, nX*nX, 1)], ['u', 'y', 'xhatp', 'Pkp'], ['xhat', 'Phat']