import numpy as np
import casadi as ca
from pynlcontrol import BasicUtils, Estimation


In, Out, InName, OutName = Estimation.KF(1, 1, 1, np.array([[-1.0]]), np.array([[1.0]]), np.array([[1.0]]), np.array([[0.1]]), np.array([[1.0]]))
KF_func = ca.Function(
    'KF_func',
    In,
    Out,
    InName,
    OutName
)

BasicUtils.Gen_Code(KF_func, 'testKF')