# Filename:     Estimator.py
# Written by:   Niranjan Bhujel
# Description:  Contains estimators such as kalman filter, extended kalman filter and simple moving horizon estimators


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

    assert isinstance(nX, int), "nX must be integer."
    assert isinstance(nU, int), "nU must be integer."
    assert isinstance(nY, int), "nY must be integer."

    assert Qw.shape[0] == Qw.shape[1], "Qw is not square matrix."
    assert Rv.shape[0] == Rv.shape[1], "Rv is not square matrix."

    assert nX == Ad.shape[0] == Ad.shape[1], "Shape mismatch of Ad with nX."
    assert nX == Bd.shape[0], "Shape mismatch of Bd with nX."
    assert nX == Cd.shape[1], "Shape mismatch of Cd with nX."
    assert nX == Qw.shape[0], "Shape mismatch of Qw with nX."

    assert nU == Bd.shape[1], "Shape mismatch of Bd with nU."
    assert nY == Cd.shape[0], "Shape mismatch of Cd with nY."
    assert nY == Rv.shape[0], "Shape mismatch of Rv with nY."

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
    assert isinstance(nX, int), "nX must be integer."
    assert isinstance(nU, int), "nU must be integer."
    assert isinstance(nY, int), "nY must be integer."

    assert Qw.shape[0] == Qw.shape[1], "Qw is not square matrix."
    assert Rv.shape[0] == Rv.shape[1], "Rv is not square matrix."

    assert nX == Qw.shape[0], "Shape mismatch of Qw with nX."

    assert nY == Rv.shape[0], "Shape mismatch of Rv with nY."

    assert isinstance(Ts, float), "Sample time (Ts) must be float."

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


def UKF(nX, nU, nY, F, H, Qw, Rv, Ts, PCoeff=None, Wm=None, Wc=None, alpha=1.0e-3, beta=2.0, kappa=0.0, Integrator='rk4'):
    """Function to implement Unscented Kalman filter. If either of PCoeff or Wm or Wc is None, it calculates those values with alpha=1e-3, Beta=2 and kappa=0.

    Args:
        nX (int): Number of state variables
        nU (int): Number of control inputs
        ny (int): Number of measurement outputs
        F (function): Function that returns right-hand side of state differential equation
        H (function): Function that retuns measurement variable from state variable
        Qw (numpy.2darray or casadi.SX array): Process noise covariance matrix
        Rv (numpy.2darray or casadi.SX array): Measurement noise covariance matrix
        Ts (float): Sample time of the Kalman filter.
        PCoeff (float): Coefficient of covariance matrix (inside square root term) when calculating sigma points. Defaults to None
        Wm (list, optional): List of weights for mean calculation. Defaults to None.
        Wc (list, optional): List of weights for covariance calculation. Defaults to None.
        alpha (float, optional): Value of alpha parameter. Defaults to 1.0e-3.
        beta (float, optional): Value of beta parameter. Defaults to 2.0.
        kappa (float, optional): Value of kappa parameter. Defaults to 0.0.
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
    def SigmaMean(S, Wm):
        N = len(Wm)
        SMean = ca.GenSX_zeros(S.shape[0], 1)
        for k in range(N):
            SMean += Wm[k] * S[:, [k]]
        return SMean


    def SigmaCovar(S1, S2, Wm, Wc):
        N = len(Wm)
        SCov = ca.GenSX_zeros(S1.shape[0], S2.shape[0])
        for k in range(N):
            SCov += Wc[k] * (S1[:, k] - SigmaMean(S1, Wm)) @ ca.transpose((S2[:, [k]] - SigmaMean(S2, Wm)))
        return SCov

    def GenSigma(xbar, P, PCoeff):
        nx = xbar.shape[0]
        S = ca.GenSX_zeros((nx, 2*nx+1))
        S[:, [0]] = xbar

        L = ca.chol(P).T

        for k in range(1, nx+1):
            S[:, [k]] = xbar + (PCoeff)**0.5 * L[:, [k-1]]
        for k in range(nx+1, 2*nx+1):
            S[:, [k]] = xbar - (PCoeff)**0.5 * L[:, [k-nx-1]]

        return S


    assert isinstance(nX, int), "nX must be integer."
    assert isinstance(nU, int), "nU must be integer."
    assert isinstance(nY, int), "nY must be integer."

    assert Qw.shape[0] == Qw.shape[1], "Qw is not square matrix."
    assert Rv.shape[0] == Rv.shape[1], "Rv is not square matrix."

    assert nX == Qw.shape[0], "Shape mismatch of Qw with nX."

    assert nY == Rv.shape[0], "Shape mismatch of Rv with nY."

    assert isinstance(Ts, float), "Sample time (Ts) must be float."
    

    if PCoeff is None or Wm is None or Wc is None:
        L = nX

        PCoeff = alpha**2*(L+kappa)

        W0m = 1-L/(alpha**2*(L+kappa))
        Wm = [W0m] + [1/(2*alpha**2*(L+kappa)) for i in range(2*nX)]

        W0c = 2-alpha**2+beta-L/(alpha**2*(L+kappa))
        Wc = [W0c] + [1/(2*alpha**2*(L+kappa)) for i in range(2*nX)]    

    xp = ca.SX.sym('xp', nX, 1)
    u = ca.SX.sym('u', nU, 1)
    y = ca.SX.sym('y', nY, 1)

    Pp = ca.SX.sym('Pp', nX*nX, 1)
    Ppt = ca.reshape(Pp, nX, nX)
    Sx = GenSigma(xp, Ppt, PCoeff)

    Sxkm = ca.GenSX_zeros(Sx.shape)

    for i in range(2*nX+1):
        Sxkm[:, i] = Integrate(F, Integrator, Ts, Sx[:, i], u)

    mukm = SigmaMean(Sxkm, Wm)
    Pkm = SigmaCovar(Sxkm, Sxkm, Wm, Wc) + Qw

    Sy = ca.GenSX_zeros(nY, 2*nX+1)

    for i in range(2*nX+1):
        Sy[:, i] = H(Sxkm[:, i])

    ykm = SigmaMean(Sy, Wm)
    Sk = SigmaCovar(Sy, Sy, Wm, Wc) + Rv

    Ck = SigmaCovar(Sxkm, Sy, Wm, Wc)

    Kk = Ck @ ca.inv(Sk)

    mu = mukm + Kk @ (y - ykm)

    Pk = Pkm - Kk @ Sk @ Kk.T

    return [u, y, xp, Pp], [mu, ca.reshape(Pk, nX*nX, 1)], ['u', 'y', 'xp', 'Pp'], ['xhat', 'Phat']


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
