from pynlcontrol import BasicUtils
import casadi as ca
import numpy as np


# Create test data
data = np.array([
    [0.0, 1, 2, 3, 4, 5, 6, 7],
    [0.5, 1, 2, 3, 4, 5, 6, 7]], dtype=np.float64)

print(data.shape)
with open("DataIn.bin", 'wb') as fw:
    data.reshape(16).tofile(fw)

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

print(Func(1, [2, 3], [4, 5, 6, 7]))

BasicUtils.Gen_Code(Func, 'GenCodeTest', dir='testGenCode',
                    mex=False, printhelp=True)

BasicUtils.Gen_Test(
    headers=['GenCodeTest.h', 'GenCodeTest_Call.h'],
    varsIn=['x', 'y', 'z'],
    sizeIn=[1, 2, 4],
    varsOut=['f1', 'f2'],
    sizeOut=[1, 1],
    callFuncName='GenCodeTest_Call_Func',
    filename='MainTest.c',
    dir='testGenCode',
)
