from pynlcontrol import Controller, BasicUtils
import numpy as np
import casadi as ca

Q = ca.SX.sym('Q')
R = ca.SX.sym('R')

In, Out, InName, OutName = Controller.LQR(
    A=np.array([[-1]]),
    B=np.array([[1]]),
    C=np.array([[1]]),
    D=np.array([[0]]),
    Q=Q,
    R=R,
    Qt=Q,
    Ts=0.05,
    horizon=1,
    reftrack=True
)

lqr_func = ca.Function(
    'lqr_func', In + [Q, R], Out, InName + ['Q', 'R'], OutName)

BasicUtils.Gen_Code(lqr_func, 'lqr_code', printhelp=True)

