# Filename:     BasicUtils.py
# Written by:   Niranjan Bhujel
# Description:  Contains basic functions such as generating interface code for casadi function,
#               numerical integration, nlp to generalized Gauss Newton (GGN), etc.


import casadi as ca
import shutil
import os
import sys
from jinja2 import Environment, FileSystemLoader


def Gen_Code(func, filename, dir='/', mex=False, printhelp=False, optim=False):
    """
    Function to generate c code (casadi generated as well as interface) for casadi function.

    Generates c and h file (from CasADi) as well as interface code. Using interface code, the generated
    codes can be integrated with other platform such as Simulink, PSIM, etc.

    Parameters
    ----------
    func : casadi.Function
        CasADi function for which code needs to be generated. This function maps input to output.
    filename : str
        File name of the generated code. Note: Filename should not contain "_Call" in it.
    dir : str, optional
        Directory where codes need to be generated. Defaults to current directory.
    mex : bool, optional
        Option if mex is required. Defaults to False.
    printhelp : bool, optional
        Option if information about input/output and its size are to be printed . If mex is False, sfunction help is also printed. Defaults to False.
    optim : bool, optional
        Whether code is being generated for CasADi optimization problem. Defaults to False.

    See Also
    --------
    Gen_Test: Generates main function code which can be compiled and debugged.

    Example
    --------
    >>> import casadi as ca
    >>> import numpy
    >>> x = ca.SX.sym('x')
    >>> y = ca.SX.sym('y', 2)
    >>> f1 = x + y[0]
    >>> f2 = x + y[1]**2
    >>> Func = ca.Function(
        'Func',
        [x, y],
        [f1, f2],
        ['x', 'y'],
        ['f1', 'f2'],
        )
    >>> BasicUtils.Gen_Code(Func, 'GenCodeTest', dir='./', mex=False, printhelp=True)
    x(1, 1), y(2, 1) -> f1(1, 1), f2(1, 1)
    GenCodeTest.c
    GenCodeTest_Call.c
    #include "GenCodeTest.h"
    #include "GenCodeTest_Call.h"
    GenCodeTest_Call_Func(x, y, f1, f2);


    Above code creates `GenCodeTest.c`, `GenCodeTest_Call.c`, `GenCodeTest.h` and `GenCodeTest_Call.h` that evaluates `f1` and `f2` from value of `x` and `y`. `x` and `y` in the above example should be declared as pointer.
    """

    funcName = func.name()

    if dir == '/' or dir == './':
        dir = ''
    else:
        if dir not in os.listdir():
            os.mkdir(dir)
        if dir[-1] != '/':
            dir += '/'

    In = [f"{x}" for x in func.name_in()]
    Out = [f"{x}" for x in func.name_out()]

    if mex:
        func.generate(f'{filename}.c', {'main': False,
                                        'mex': True, 'with_header': True})
        shutil.move(f'{filename}.c', f'{dir}{filename}.c')
        shutil.move(f'{filename}.h', f'{dir}{filename}.h')
    else:
        func.generate(f'{filename}.c', {'main': False,
                                        'mex': False, 'with_header': True})
        shutil.move(f'{filename}.c', f'{dir}{filename}.c')
        shutil.move(f'{filename}.h', f'{dir}{filename}.h')

        SizeIn = []
        for k in range(len(In)):
            SizeIn.append(func.size_in(k)[0]*func.size_in(k)[1])

        SizeOut = []
        for k in range(len(Out)):
            SizeOut.append(func.size_out(k)[0]*func.size_out(k)[1])

        totalIOsize = 0
        for k in range(len(In)):
            totalIOsize += func.size_in(k)[0]*func.size_in(k)[1]
        for k in range(len(Out)):
            totalIOsize += func.size_out(k)[0]*func.size_out(k)[1]

        # template rendering
        templatePath = f"{os.path.dirname(os.path.realpath(__file__))}/templates"

        file_loader = FileSystemLoader(templatePath)
        env = Environment(loader=file_loader)

        # template for c file
        template = env.get_template('GenCode_c.j2')

        CodeOutput_c = template.render(
            FileName=filename,
            Func_Name=funcName,
            Func_In=In,
            Func_Out=Out,
            arg_size=func.sz_arg(),
            res_size=func.sz_res(),
            iw_size=func.sz_iw(),
            w_size=int(optim)*func.sz_w() + totalIOsize,
            SizeIn=SizeIn,
            SizeOut=SizeOut,
        )

        # template for h file
        template = env.get_template('GenCode_h.j2')

        CodeOutput_h = template.render(
            FileName=filename,
            Func_Name=funcName,
            Func_In=In,
            Func_Out=Out,
            arg_size=func.sz_arg(),
            res_size=func.sz_res(),
            iw_size=func.sz_iw(),
            w_size=func.sz_w(),
            SizeIn=SizeIn,
            SizeOut=SizeOut,
        )

        # Write code to file
        with open(f"{dir}{filename}_Call.c", 'w') as fw:
            fw.write(CodeOutput_c)

        with open(f"{dir}{filename}_Call.h", 'w') as fw:
            fw.write(CodeOutput_h)

    # print information about input/out and its size.
    if printhelp:
        tmpIn = []
        for k in range(len(In)):
            tmpIn.append(
                f"{In[k]}({func.size_in(k)[0]}, {func.size_in(k)[1]})")

        tmpOut = []
        for k in range(len(Out)):
            tmpOut.append(
                f"{Out[k]}({func.size_out(k)[0]}, {func.size_out(k)[1]})")
        print(", ".join(tmpIn) + " -> " + ", ".join(tmpOut))
        print("\n")

        if ~mex:
            print(f"{dir}{filename}.c\n{dir}{filename}_Call.c\n\n")
            print(
                f'#include "{dir}{filename}.h"\n#include "{dir}{filename}_Call.h"\n\n')
            print(f'{filename}_Call_Func({", ".join(In)}, {", ".join(Out)});\n')


def Gen_Test(headers, varsIn, sizeIn, varsOut, sizeOut, callFuncName, filename, dir='/'):
    """
    Generates C code with main function. This code along with code generated by `Gen_Code` function can be compiled to executable to check computation time or debug.
    It can also be used as example to compile on other target.

    Parameters
    ----------
    headers : list[str]
        List of header files name along with extension `.h`. These files will be added in main file as #include "header".
    varsIn : list[str]
        List of name of input variables.
    sizeIn : list[int]
        List of size of input variables.
    varsOut : list(str)
        List of name of output variables.
    sizeOut : list[int]
        List of size of output variables.
    callFuncName : list(str)
        Name of the C function that needs to be called.
    filename : str
        Name of the generated C file (along with extension .c).
    dir : str, optional
        Directory where code needs to be generated. Defaults to current directory.

    Example
    _______
    >>> from pynlcontrol import BasicUtils
    >>> import casadi as ca
    >>> x = ca.SX.sym('x')
    >>> y = ca.SX.sym('y', 2)
    >>> f1 = x + y[0]
    >>> f2 = x + y[1]**2
    >>> Func = ca.Function(
        'Func',
        [x, y],
        [f1, f2],
        ['x', 'y'],
        ['f1', 'f2'],
        )
    >>> BasicUtils.Gen_Code(Func, 'GenCodeTest', dir='./', mex=False, printhelp=False)
    >>> BasicUtils.Gen_Test(
        headers=['GenCodeTest.h', 'GenCodeTest_Call.h'],
        varsIn=['x', 'y'],
        sizeIn=[1, 2],
        varsOut=['f1', 'f2'],
        sizeOut=[1, 1],
        callFuncName='GenCodeTest_Call_Func',
        filename='MainTest.c',
        dir='./',
        )

    
    All the line except last one is same as from `Gen_Code()` function. The last line generates filename with `MainTest.c` which can be used to test and debug the generated code.
    """



    if dir == '/' or dir == './':
        dir = ''
    else:
        if dir not in os.listdir():
            os.mkdir(dir)
        if dir[-1] != '/':
            dir += '/'

    templatePath = f"{os.path.dirname(os.path.realpath(__file__))}/templates"

    file_loader = FileSystemLoader(templatePath)
    env = Environment(loader=file_loader)

    # template for c file
    template = env.get_template('TestCode.j2')

    CodeOutput = template.render(
        headers=headers,
        sizeIn=sizeIn,
        varsIn=varsIn,
        sizeOut=sizeOut,
        varsOut=varsOut,
        callFuncName=callFuncName
    )

    with open(f"{dir}{filename}", 'w') as fw:
        fw.write(CodeOutput)


def Integrate(odefun, method, Ts, x0, u0, *args):
    """
    Function to integrate continuous-time ODE. It discretize the provided ODE function and gives value of state variables at next discrete time. 

    Parameters
    ----------
    odefun : function
        Python function with states as first and control input as second argument. Remaining argument could be anything required by state equations.
    method : str
        Method to integrate ODE. Supported methods are: 'FEuler', 'rk2', 'rk3', 'ssprk3', 'rk4', 'dormandprince'.
    Ts : float
        Step time to solve ODE
    x0 : float or casadi.SX or numpy.1darray
        Current states of the system
    u0 : float or casadi.SX or numpy.1darray
        Current control input to the system

    Returns
    -------
    float or casadi.SX or numpy.1darray
        Next states of the system.

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> def Fc(x, u):
        x1 = x[0]
        x2 = x[1]
        return np.array([(1-x2**2)*x1 - x2 + u, x1])
    >>> T = 10
    >>> Ts = 0.1
    >>> t = np.arange(0, T+Ts, Ts)
    >>> x = np.zeros((2, t.shape[0]))
    >>> for k in range(t.shape[0]-1):
        u = 0 if t[k] < 1 else 1.0
        x[:,k+1] = Integrate(Fc, 'rk4', Ts, x[:,k], u)
    >>> plt.plot(t, x[0,:])
    >>> plt.plot(t, x[1,:])
    >>> plt.xlabel('time (s)')
    >>> plt.ylabel('$x_1$ and $x_2$')
    >>> plt.show()

    The above code integrates the ODE defined by function Fc from 0 to 10 s with step-time of 0.1 s using RK4 method.
    """
    if method == 'FEuler':
        xf = x0 + Ts*odefun(x0, u0, *args)
    elif method == 'rk2':
        k1 = odefun(x0, u0, *args)
        k2 = odefun(x0 + Ts/2*k1, u0, *args)
        xf = x0 + Ts*k2
    elif method == 'rk3':
        k1 = odefun(x0, u0, *args)
        k2 = odefun(x0+Ts/2*k1, u0, *args)
        k3 = odefun(x0+Ts*k2, u0, *args)
        xf = x0 + Ts/6*(k1 + 4*k2 + k3)
    elif method == 'rk4':
        k1 = odefun(x0, u0, *args)
        k2 = odefun(x0+Ts/2*k1, u0, *args)
        k3 = odefun(x0+Ts/2*k2, u0, *args)
        k4 = odefun(x0 + Ts*k3, u0, *args)
        xf = x0 + Ts/6*(k1 + 2*k2 + 2*k3 + k4)
    elif method == 'ssprk3':
        k1 = odefun(x0, u0, *args)
        k2 = odefun(x0+Ts*k1, u0, *args)
        k3 = odefun(x0+Ts/2*k2, u0, *args)
        xf = x0 + Ts/6*(k1 + k2 + 4*k3)
    elif method == 'dormandprince':
        k1 = odefun(x0, u0, *args)
        k2 = odefun(x0 + 1/5*Ts*k1, u0, *args)
        k3 = odefun(x0 + 3/10*Ts*k2, u0, *args)
        k4 = odefun(x0 + 4/5*Ts*k3, u0, *args)
        k5 = odefun(x0 + 8/9*Ts*k4, u0, *args)
        k6 = odefun(x0 + 1*Ts*k5, u0, *args)
        k7 = odefun(x0 + 1*Ts*k6, u0, *args)
        # Xf = x0 + Ts*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6)
        xf = x0 + Ts*(5179/57600*k1 + 7571/16695*k3 + 393 /
                      640*k4 - 92097/339200*k5 + 187/2100*k6 + 1/40*k7)
    else:
        print(f'Error: No integrator method {method} found.')
        sys.exit(1)
    return xf


def nlp2GGN(z, J, g, lbg, ubg, p):
    """
    Converts provided nonlinear programming into quadratic form using generalized Gauss-Newton method.

    Parameters
    ----------
    z : casadi.SX
        Vector of unknown variables of optimization problem
    J : casadi.SX
        Object function of the optimization problem
    g : casadi.SX
        Vector of constraints function
    lbg : casadi.SX
        Vector of lower limits on constraint function
    ubg : casadi.SX
        Vector of upper limits on constraint function
    p : casadi.SX
        Vector of input to the optimization problem

    Returns
    -------
    dict
        Dictionary of optimization problem.
        keywords
            x: Vector of decision variables 
            f: New quadratic cost function
            g: New constraint function
            lbg: Lower limits on constraint function
            ubg: Upper limits on constraint function
    """
    zOp = ca.SX.sym('zOp', z.shape)

    Jnew = ca.substitute(J, z, zOp) + \
        ca.substitute(ca.jacobian(J, z), z, zOp) @ (z - zOp)

    Cost = ca.norm_2(Jnew)**2

    nlp = {
        'x': z,
        'f': Cost
    }

    if g is not None:
        gnew = ca.substitute(g, z, zOp) + \
            ca.substitute(ca.jacobian(g, z), z, zOp) @ (z - zOp)
        nlp['g'] = gnew
        nlp['lbg'] = lbg
        nlp['ubg'] = ubg

    if p is not None:
        nlp['p'] = p

    nlp['zOp'] = zOp

    return nlp


def __SXname__(x):
    """
    Returns the name of casadi.SX symbolics

    Parameters
    ----------
    x : list[casadi.SX] or casadi.SX
        List of casadi symbolics or just casadi symbolics.

    Returns
    -------
    list[str] or str
        List of name of symbolics or just name of symbolics
    """
    if isinstance(x, list):
        nameList = []
        for tmp in x:
            z = str(tmp)
            z = z.split('_')
            nameList.append(z[0])

        return nameList
    else:
        z = str(x[0])
        z = z.split('_')
        return z[0]


def directSum(A):
    """
    Direct sum of matrices in the list A.

    Parameters
    ----------
    A : list
        List of matrices.

    Returns
    -------
    casadi.SX.sym
        Direct sum of all matrices in list A.

    Example
    -------
    >>> import casadi as ca
    >>> from pynlcontrol import BasicUtils
    >>> import casadi as ca
    >>> from pynlcontrol import BasicUtils
    >>> A1 = ca.SX.sym('A1', 2, 2)
    >>> A2 = ca.SX.sym('A2', 3, 3)
    >>> A = [A1, A2]
    >>> BasicUtils.directSum(A)
    SX(@1=0,
    [[A1_0, A1_2, @1, @1, @1],
     [A1_1, A1_3, @1, @1, @1],
     [@1, @1, A2_0, A2_3, A2_6],
     [@1, @1, A2_1, A2_4, A2_7],
     [@1, @1, A2_2, A2_5, A2_8]])

    Above code puts matrices `A1` and `A2` in the diagonal and fills zero elsewhere.
    """

    assert isinstance(A, list), "Argument must be list."
    N = len(A)
    NRow = 0
    NCol = 0
    for i in range(N):
        NRow += A[i].shape[0]
        NCol += A[i].shape[1]
    z = ca.GenSX_zeros((NRow, NCol))
    Row_Idx = 0
    Col_Idx = 0
    for i in range(N):
        z[Row_Idx:Row_Idx+A[i].shape[0], Col_Idx:Col_Idx+A[i].shape[1]] = A[i]
        Row_Idx += A[i].shape[0]
        Col_Idx += A[i].shape[1]
    return z


def casadi2List(x):
    n = x.shape[0]
    z = []
    for k in range(n):
        z.append(float(x[k]))
    return z



def qrSym(A):
    """
    Performs QR decomposition of matrix A using householder reflection. 

    Args:
        A (ca.SX.sym): Matrix A in casadi symbolics

    Returns:
        Q: Q matrix of decomposition
        R: R matrix of decomposition
    """

    
    m, n = A.shape

    Q = ca.SX_eye(m)

    R = ca.GenSX_zeros(m, n)

    for i in range(m):
        for j in range(n):
            R[i,j] = A[i,j]

    for k in range(n):
        x = R[k:,[k]]

        alpha = ca.norm_2(x)
        e = ca.GenSX_zeros(m-k, 1)
        e[0, 0] = 1.0

        u = x - alpha * e

        v = u/ca.norm_2(u)

        Qtmp = ca.SX_eye(m)

        Qtmp[k:,k:] = ca.SX_eye(m-k) - 2*v@v.T

        Q = Q @ Qtmp
        R = Qtmp @ R

    return Q, R
