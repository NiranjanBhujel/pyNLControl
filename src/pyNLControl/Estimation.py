from pynlcontrol.BasicUtils import Integrate, nlp2GGN
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


def EKF(nX, nU, nY, F, H, Qw, Rv, Ts, Integrator='rk4'):
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
    y = ca.SX.sym('y', nY, 1)

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


def simpleMHE(nX, nU, nY, nP, N, Fc, Hc, Wp, Wm, Ts, pLower=[], pUpper=[], arrival=False, GGN=False, Integrator='rk4', Options=None):
    """Function to generate simple MHE code using `qrqp` solver. For use with other advanced solver, see `MPC` class.

    Args:
        nX (int): Number of state variables.
        nU (int): number of control input.
        nY (int): Number of measurement variables.
        nP (int): Number of parameter to be estimated. nP=0 while performing state estimation only.
        N (int): Horizon length.
        Fc (function): Function that returns right hand side of state equation.
        Hc (function): Function that returns right hand side of measurement equation.
        Wp (float or casadi.SX array or numpy.2darray): Weight for process noise term. It is $Q_w^{-1/2}$ where $Q_w$ is process noise covariance.
        Wm (float or casadi.SX array or numpy.2darray): Weight for measurement noise term. It is $R_v^{-1/2}$ where $R_v$ is measurement noise covariance.
        Ts (float): Sample time for MHE
        pLower (list, optional): List of lower limits of unknown parameters. Defaults to [].
        pUpper (list, optional): List of upper limits of unknown parameters. Defaults to [].
        arrival (bool, optional): Whether to include arrival cost. Defaults to False.
        GGN (bool, optional): Whether to use GGN. Use this option only when optimization problem is nonlinear. Defaults to False.
        Integrator (str, optional): Integration method. See `BasicUtils.Integrate()` function. Defaults to 'rk4'.
        Options (dict, optional): Option for `qrqp` solver. Defaults to None.

    Returns:
        tuple: tuple: Tuple of Input, Output, Input name and Output name. Input and output are list of casadi symbolics (`casadi.SX`).
            Input should be control input and measurement data of past horizon length
            Output are all value of decision variable, estimations of parameter, estimates of states and cost function.
    """

    estParam = nP > 0

    X = ca.SX.sym('X', nX, N+1)
    U = ca.SX.sym('U', nU, N)
    Y = ca.SX.sym('Y', nY, N+1)

    J = ca.vertcat()

    # Symbolics for parameters
    if estParam:
        P = ca.SX.sym('P', nP, 1)

    # Arrival cost in cost function formulation
    if arrival:
        xb = ca.SX.sym('xb', nX, 1)
        Wax = ca.SX.sym('Wax', nX*nX, 1)
        J = ca.vertcat(
            J,
            Wax.reshape((nX, nX)) @ (X[:, 0] - xb)
        )
        # Add parameter as well in arrival cost
        if estParam:
            pb = ca.SX.sym('pb', nP, 1)
            Wap = ca.SX.sym('Wap', nP*nP, 1)
            J = ca.vertcat(
                J,
                Wap.reshape((nP, nP)) @ (P - pb)
            )

    # Stage terms in cost function
    for k in range(N):
        J = ca.vertcat(
            J,
            Wm @ (Y[:, k] - Hc(X[:, k])),
            Wp @ (X[:, k+1] - Integrate(Fc, Integrator, Ts, X[:, k], U[:, k], P))
        )

    J = ca.vertcat(
        J,
        Wm @ (Y[:, N] - Hc(X[:, N]))
    )

    # Constraints on parameters
    if estParam:
        g = ca.vertcat()
        lbg = ca.vertcat()
        ubg = ca.vertcat()

        for k in range(nP):
            g = ca.vertcat(
                g,
                P[k]
            )
            lbg = ca.vertcat(
                lbg,
                pLower[k]
            )
            ubg = ca.vertcat(
                ubg,
                pUpper[k]
            )
    else:
        g = None
        lbg = None
        ubg = None

    # decision variables
    if estParam:
        z = ca.vertcat(
            P,
            X.reshape((-1, 1))
        )
    else:
        z = X.reshape((-1, 1))

    pIn = ca.vertcat(
        U.reshape((-1, 1)),
        Y.reshape((-1, 1))
    )

    if arrival:
        pIn = ca.vertcat(
            pIn,
            xb,
            Wax
        )

        if estParam:
            pIn = ca.vertcat(
                pIn,
                pb,
                Wap
            )

    if GGN:
        nlp = nlp2GGN(z, J, g, lbg, ubg, pIn)
        nlp['p'] = ca.vertcat(
            nlp['p'],
            nlp['zOp']
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

    MHE_prob = {
        'x': nlp['x'],
        'f': nlp['f'],
        'g': nlp['g'],
        'p': nlp['p']
    }

    S = ca.nlpsol('S', 'sqpmethod', MHE_prob, Options)

    xGuess = ca.MX.sym('xGuess', MHE_prob['x'].shape)
    Up = ca.MX.sym('Up', nU, N)
    Yp = ca.MX.sym('Yp', nY, N+1)

    pVal = ca.vertcat(
        Up.reshape((-1, 1)),
        Yp.reshape((-1, 1))
    )

    if arrival:
        xbp = ca.MX.sym('xbp', xb.shape)
        Waxp = ca.MX.sym('Waxp', nX*nX, 1)
        pVal = ca.vertcat(
            pVal,
            xbp,
            Waxp
        )
        if estParam:
            pbp = ca.MX.sym('pbp', nP, 1)
            Wapp = ca.MX.sym('Wapp', nP*nP, 1)
            pVal = ca.vertcat(
                pVal,
                pbp,
                Wapp
            )

    if GGN:
        zOpp = ca.MX.sym('zOpp', z.shape)
        pVal = ca.vertcat(
            pVal,
            zOpp
        )

    if estParam:
        r = S(x0=xGuess, p=pVal, lbg=lbg, ubg=ubg)
    else:
        r = S(x0=xGuess, p=pVal)

    In = [
        xGuess,
        Up,
        Yp,
    ]
    InName = [
        'zGuess',
        'um',
        'ym',
    ]

    if arrival:
        In += [xbp, Waxp]
        InName += ['xL', 'WxL']
        if estParam:
            In += [pbp, Wapp]
            InName += ['pL', 'WpL']

    if GGN:
        In.append(zOpp)
        InName.append('zOp')

    Out = [r['x']]
    OutName = ['zOut']

    if estParam:
        Out.append(r['x'][0:nP])
        OutName.append('p_hat')

    Out.append(r['x'][nP+N*nX:nP+(N+1)*nX])
    OutName.append('x_hat')

    Out.append(r['f'])
    OutName.append('Cost')

    return In, Out, InName, OutName
