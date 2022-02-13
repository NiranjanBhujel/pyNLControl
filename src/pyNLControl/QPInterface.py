# Filename:     QPInterface.py
# Written by:   Niranjan Bhujel
# Description:  Contains interface to qp solver. Currently, only qpOASES is supported.


from pynlcontrol import BasicUtils
import casadi as ca
import numpy
import os
from jinja2 import Environment, FileSystemLoader
import zipfile


class qpOASES:
    def __init__(self, H, g, p=None, A=None, lbA=None, ubA=None, lbx=None, ubx=None) -> None:
        """Class to create interface to qpOASES solver.

        Args:
            H (casadi.SX): Hessian matrix of cost function
            g (casadi.SX): Linear coefficient vector of cost function
            p (list, optional): List of input parameters to optimization problem. Defaults to None.
            A (casadi.SX, optional): Constraint matrix. Defaults to None.
            lbA (casadi.SX, optional): Lower bound on constraint matrix. Defaults to None.
            ubA (casadi.SX, optional): Upper bound on constraint matrix. Defaults to None.
            lbx (casadi.SX, optional): Lower bound on decision variables. Defaults to None.
            ubx (casadi.SX, optional): Upper bound on decision variables. Defaults to None.
        """
        self.H = ca.densify(H)
        self.g = g

        self.OutName = ["H", "g"]
        self.Out = [self.H.T, self.g]
        self.OutSize = [
            f"{self.H.shape[0]}*{self.H.shape[1]}",
            f"{self.g.shape[0]}"
        ]

        self.In = []
        self.InName = []

        if p is not None:
            self.p = p
            self.In += self.p
            self.InName += BasicUtils.__SXname__(self.p)

        if A is not None:
            self.A = ca.densify(A)
            self.OutName.append("A")
            self.Out.append(self.A.T)
            self.OutSize.append(f"{self.A.shape[0]}*{self.A.shape[1]}")

            self.lbA = lbA if lbA is not None else - \
                ca.inf*numpy.ones((self.A.shape[0]))
            self.OutName.append("lbA")
            self.Out.append(self.lbA)
            self.OutSize.append(f"{self.lbA.shape[0]}")

            self.ubA = ubA if ubA is not None else ca.inf * \
                numpy.ones((self.A.shape[0]))
            self.OutName.append("ubA")
            self.Out.append(self.ubA)
            self.OutSize.append(f"{self.ubA.shape[0]}")

        self.lbx = lbx if lbx is not None else -ca.inf*numpy.ones((H.shape[0]))
        self.OutName.append("lbx")
        self.Out.append(self.lbx)
        self.OutSize.append(f"{self.lbx.shape[0]}")

        self.ubx = ubx if ubx is not None else ca.inf*numpy.ones((H.shape[0]))
        self.OutName.append("ubx")
        self.Out.append(self.ubx)
        self.OutSize.append(f"{self.lbx.shape[0]}")

        self.nV = H.shape[0]
        if A is not None:
            self.nC = lbA.shape[0]
        else:
            self.nC = 0

    def exportEvalCode(self, funcname, dir='/', options=None):
        """Method of class qpOASES to generate C code to evaluate H, h, A, lbA, ubA, lbx, ubx.

        Args:
            funcname (str): Function to be named that evaluates H, h, A, lbA, ubA, lbx and ubx.
            dir (str, optional): Directory where codes are to be exported. Defaults to '/'.
            options (dict, optional): Options for code generation. Same option as casadi.Function.generate() function. Defaults to None.
        """
        if options is None:
            options = {
                'main': False,
                'with_header': True
            }

        Func = ca.Function(
            funcname,
            self.In,
            self.Out,
            self.InName,
            self.OutName
        )

        BasicUtils.Gen_Code(Func, funcname+'_CODE', dir, printhelp=False)

    def exportCode(self, filename, dir='/', mex=False, printsfun=False, Options=None, TestCode=False):
        """Method of class qpOASES to export the code that solves quadratic programming.

        Args:
            filename (str): Filename for exported code.
            dir (str, optional): Directory where code is to be exported. Defaults to '/'.
            mex (bool, optional): Whether mex interface is required. Defaults to False.
            printsfun (bool, optional): Whether MATLAB s-function interface is to be implemented. Defaults to False.
            Options (dict, optional): Options for qpOASES solver. Defaults to None.
            TestCode (bool, optional): Whether main function is required. Useful while testing and debugging. Defaults to False.
        """

        if Options is not None:
            if "max_iter" in Options:
                MaxIter = Options['max_iter']
            else:
                MaxIter = 10
            Options.pop('max_iter', None)
            Opt = Options
        else:
            Opt = {}
            MaxIter = 10

        if dir[-1] != '/':
            tmpdir = dir + '/'
        else:
            tmpdir = dir[0:-1]

        with zipfile.ZipFile(f"{os.path.dirname(os.path.realpath(__file__))}\\external\\qpOASES.zip", 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        self.exportEvalCode(filename + "_EVAL", dir)

        OutList = [
            'A',
            'lbx',
            'ubx',
            'lbA',
            'ubA'
        ]
        OutPointer = []
        for tmp in OutList:
            if tmp in self.OutName:
                OutPointer.append(tmp)
            else:
                OutPointer.append("NULL")

        # template rendering
        templatePath = f"{os.path.dirname(os.path.realpath(__file__))}/templates"

        file_loader = FileSystemLoader(templatePath)
        env = Environment(loader=file_loader)

        # template for c file
        template = env.get_template('qpOASES_c.j2')

        CodeOutput_c = template.render(
            FileName=filename,
            InName=self.InName,
            OutName=self.OutName,
            OutSize=self.OutSize,
            nV=self.nV,
            nC=self.nC,
            OutPointer=OutPointer,
            Options=Opt,
            MaxIter=MaxIter,
            Mex=mex
        )

        template = env.get_template('qpOASES_h.j2')

        CodeOutput_h = template.render(
            FileName=filename,
            InName=self.InName,
            OutName=self.OutName,
            OutSize=self.OutSize,
            nV=self.nV,
            nC=self.nC,
            OutPointer=OutPointer,
            Options=Opt,
            MaxIter=MaxIter,
            Mex=mex
        )

        if mex:
            template = env.get_template('Gen_Mex_c.j2')

            MexOutput_c = template.render(
                InNum=len(self.In)+1,
                InName=["xGuess"] + self.InName,
                OutNum=2,
                OutName=["xOpt", "Obj_VAL"],
                OutSize=[self.H.shape[0], 1],
                FuncCall=f"{filename}_Call"
            )

            CodeOutput_c += "\n\n" + MexOutput_c

        with open(f"{tmpdir}{filename}.c", "w") as fw:
            fw.write(CodeOutput_c)

        with open(f"{tmpdir}{filename}.h", "w") as fw:
            fw.write(CodeOutput_h)

        if mex == True:
            Fin = [f'xGuess[{self.H.shape[0]}x1]']
            for k in range(len(self.InName)):
                Fin.append(
                    str(self.InName[k]) + f'[{self.In[k].shape[0]}x{self.In[k].shape[1]}]')
            Fin = ', '.join(Fin)
            Fout = [f'xOpt_VAL[{self.H.shape[0]}x{1}]', 'Obj_VAL[1x1]']
            Fout = ', '.join(Fout)

            # Print inputs and outputs with their respective size
            print(f'\n({Fin})->({Fout})\n')
        if printsfun and mex == False:
            Fin = [f'xGuess[{self.H.shape[0]}x1]']
            for k in range(len(self.InName)):
                Fin.append(
                    str(self.InName[k]) + f'[{self.In[k].shape[0]}x{self.In[k].shape[1]}]')
            Fin = ', '.join(Fin)
            Fout = [f'xOpt_VAL[{self.H.shape[0]}x{1}]', 'Obj_VAL[1x1]']
            Fout = ', '.join(Fout)

            # Print inputs and outputs with their respective size
            print(f'\n({Fin})->({Fout})\n')

            FuncIn_Str = ', '.join(self.InName)
            # print('INC_PATH qpOASES/include')
            print(f'{tmpdir}{filename}.c')
            print(f'{tmpdir}{filename}_EVAL_CODE.c')
            print(f'{tmpdir}{filename}_EVAL_CODE_Call.c')
            print(f"{tmpdir}BLASReplacement.cpp")
            print(f"{tmpdir}Bounds.cpp")
            print(f"{tmpdir}Constraints.cpp")
            print(f"{tmpdir}Flipper.cpp")
            print(f"{tmpdir}Indexlist.cpp")
            print(f"{tmpdir}LAPACKReplacement.cpp")
            print(f"{tmpdir}Matrices.cpp")
            print(f"{tmpdir}MessageHandling.cpp")
            print(f"{tmpdir}Options.cpp")
            print(f"{tmpdir}OQPinterface.cpp")
            print(f"{tmpdir}QProblem.cpp")
            print(f"{tmpdir}QProblemB.cpp")
            print(f"{tmpdir}SolutionAnalysis.cpp")
            print(f"{tmpdir}SparseSolver.cpp")
            print(f"{tmpdir}SQProblem.cpp")
            print(f"{tmpdir}SQProblemSchur.cpp")
            print(f"{tmpdir}SubjectTo.cpp")
            print(f"{tmpdir}Utils.cpp")

            print('\n')
            print(f'#include "{tmpdir}{filename}.h"')
            print(f'#include "{tmpdir}{filename}_EVAL_CODE.h"')
            print(f'#include "{tmpdir}{filename}_EVAL_CODE_Call.h"\n')

            print('\n# Try this first:')
            print(f'{filename}_Call(xGuess, {FuncIn_Str}, xOpt_VAL, Obj_VAL);')
            print('\n# If previous does not work, try this:')
            print(f"double *xGuess_TEMP = (double *)xGuess;")
            for k in range(len(self.InName)):
                print(
                    f"double *{self.InName[k]}_TEMP = (double *){self.InName[k]};")

            FuncIn = []
            for tmp in self.InName:
                FuncIn.append(str(tmp) + '_TEMP')
            FuncIn_Str = ', '.join(FuncIn)
            print(f'{filename}_Call(xGuess_TEMP, {FuncIn_Str}, xOpt_VAL, Obj_VAL);\n')

        if TestCode:
            Headers = [f"{filename}.h", f"{filename}_EVAL_CODE.h",
                       f"{filename}_EVAL_CODE_Call.h"]
            VarsIn = ['xGuess'] + self.InName
            SizeIn = [self.H.shape[0]] + [self.In[k].shape[0] *
                                          self.In[k].shape[1] for k in range(len(self.In))]
            VarsOut = ['xOpt_VAL', 'Obj_VAL']
            SizeOut = [self.H.shape[0], 1]
            Call_FuncName = f'{filename}_Call'

            BasicUtils.Gen_Test(
                Headers,
                VarsIn,
                SizeIn,
                VarsOut,
                SizeOut,
                Call_FuncName,
                'Main_'+filename+'.c',
                dir
            )
