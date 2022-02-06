import casadi as ca
import shutil
import os
import sys
from jinja2 import Environment, FileSystemLoader


def Gen_Code(func, filename, dir='/', mex=False, printhelp=False, optim=False):
    """Function to generate c code (casadi generated as well as interface) for casadi function.

    Args:
        func (casadi.Function): CasADi function for which code needs to be generated
        filename (str): Name of the generated code
        dir (str, optional): Directory where codes need to be generated. Defaults to current directory.
        mex (bool, optional): Option if mex is required. Defaults to False.
        printhelp (bool, optional): Option if information about input/output and its size are to be printed . If mex is False, sfunction help is also printed. Defaults to False.
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
            w_size=func.sz_w() + int(optim)*totalIOsize,
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
    """Generates C code with main function. This code along with code generated by `Gen_Code` function can be compiled to executable to check computation time or debug.
    It can also be used as example to compile on other target.

    Args:
        headers (list[str]): List of header files name along with extension .h). These files will be added in main file as #include "header".
        varsIn (list[str]): List of name of input variables.
        sizeIn (list[int]): List of size of input variables.
        varsOut (list(str)): List of name of output variables.
        sizeOut (list(int)): List of size of output variables.
        callFuncName (str): Name of the C function that needs to be called.
        filename (str): Name of the generated C file (along with extension .c).
        dir ([str], optional): Directory where code needs to be generated. Defaults to current directory.
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
    """Function to integrate continuous-time ODE. It discretize the provided ODE function and gives value of state variables at next discrete time. 

    Args:
        odefun (function): Python function with states as first and control input as second argument. Remaining argument could be anything required by state equations
        method (str): Method to integrate ODE. Supported methods are: 'FEuler', 'rk2', 'rk3', 'ssprk3', 'rk4', 'dormandprince'.
        Ts (float): Step time to solve ODE
        x0 (float or casadi.SX or numpy.1darray): Current states of the system
        u0 (float or casadi.SX or numpy.1darray): Current control input to the system

    Returns:
        float or casadi.SX or numpy.1darray: Next states of the system
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
    """Converts provided nonlinear programming into quadratic form using generalized Gauss-Newton method.

    Args:
        z (casadi.SX): Vector of unknown variables of optimization problem
        J (casadi.SX): Object function of the optimization problem
        g (casadi.SX): Vector of constraints function
        lbg (casadi.SX): Vector of lower limits on constraint function
        ubg (casadi.SX): Vector of upper limits on constraint function
        p (casadi.SX): Vector of input to the optimization problem

    Returns:
        dict: Dictionary of optimization problem.
              keywords
              x: Vector of decision variables 
              f: New quadratic cost function
              g: New constraint function
              lbg: Lower limits on constraint function
              ubg: Upper limits on constraint function

    """
    zOp = ca.SX.sym('zOp', z.shape)

    Jnew = ca.substitute(J, z, zOp) + ca.substitute(ca.jacobian(J, z), z, zOp) @ (z - zOp)

    Cost = ca.norm_2(Jnew)**2

    nlp = {
        'x': z,
        'f': Cost
    }
    
    if g is not None:
        gnew = ca.substitute(g, z, zOp) + ca.substitute(ca.jacobian(g, z), z, zOp) @ (z - zOp)
        nlp['g'] = gnew
        nlp['lbg'] = lbg
        nlp['ubg'] = ubg
        
    if p is not None:
        nlp['p'] = p

    nlp['zOp'] = zOp

    return nlp
