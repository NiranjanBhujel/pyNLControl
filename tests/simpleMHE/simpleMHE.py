from pynlcontrol import Estimation, BasicUtils
import casadi as ca

def Fc(x, u, p):
    R = 1
    L = p
    return -R/L*x + 1/L*u

def Hc(x):
    return x

In, Out, InName, OutName = Estimation.simpleMHE(
    nX=1,
    nU=1,
    nY=1,
    nP=1,
    N=12,
    Fc=Fc,
    Hc=Hc,
    Wp=1,
    Wm=0.3,
    Ts=0.005,
    pLower=[0],
    pUpper=[0.2],
    GGN=True
)

MHE_func = ca.Function(
    'MHE_func',
    In,
    Out,
    InName,
    OutName
)


BasicUtils.Gen_Code(
    MHE_func,
    'MHE_Code',
    printhelp=True,
    optim=True
)
