from pyNLControl import BasicUtils
import casadi as ca

x = ca.SX.sym('x')
y = ca.SX.sym('y', 2)
z = ca.SX.sym('z', 4)

f1 = x**2 + z[0]
f2 = y[0] + y[1]

Func = ca.Function(
    'Func',
    [x, y, z],
    [f1, f2],
    ['x', 'y', 'z'],
    ['f1', 'f2'])

BasicUtils.Gen_Code(Func, 'GenCodeTest', dir='testGencode',
                    mex=False, printhelp=True)
