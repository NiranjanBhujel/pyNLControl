import casadi as ca
from pynlcontrol import BasicUtils, Controller


def Fc(x, u):
    A = ca.SX(
        [
            [-0.4, 0.1, -2],
            [0, -0.3, 4],
            [1, 0, 0]
        ]
    )
    B = ca.SX(
        [
            [1, 1],
            [0, 1],
            [1, 0]
        ]
    )
    return A @ x + B @ u


def Hc(x):
    return ca.vertcat(x[0], x[1])


In, Out, InName, OutName = Controller.simpleMPC(nX=3, nU=2, nY=2, Fc=Fc, Fc=Hc, N=25, Ts=0.1, uLow=[-10, 0], uUpp=[10, 3], GGN=False)


MPC_func = ca.Function('MPC_func', In, Out, InName, OutName)

BasicUtils.Gen_Code(MPC_func, 'MPC_Code', printhelp=True, optim=True)
