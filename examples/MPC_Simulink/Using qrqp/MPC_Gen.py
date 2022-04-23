import casadi as ca
from pynlcontrol import BasicUtils, Controller


def Fc(x, u, p):
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


def Hc(x, p):
    return ca.vertcat(x[0], x[1])

def Gk(x, u, p):
    return -0.1, u[0] - x[0], 0.1



In, Out, InName, OutName = Controller.simpleMPC(3, 2, 2, 0, Fc, Hc, None, None, 25, 0.1, [-10, 0], [10, 3], GGN=False)


MPC_func = ca.Function('MPC_func', In, Out, InName, OutName)

BasicUtils.Gen_Code(MPC_func, 'MPC_Code', printhelp=True, optim=True)
