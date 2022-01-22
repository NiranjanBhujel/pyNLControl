import casadi as ca
import shutil
import os
from jinja2 import Environment, FileSystemLoader


def Gen_Code(func, dir, filename, mex=False, printhelp=False):
    funcName = func.name()

    if dir=='/' or dir=='./':
        dir = ''
    else:
        if dir[-1] != '/':
            dir += '/'

    In = [f"{x}" for x in func.name_in()]
    Out = [f"{x}" for x in func.name_out()]

    if mex:
        func.generate(f'{filename}.c', {'main': False,
                                        'mex': True, 'with_header': False})
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
            w_size=func.sz_w(),
            SizeIn=SizeIn,
            SizeOut=SizeOut,
        )

        # template for c file
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

    if printhelp:
        tmp1 = []
        for k in range(len(In)):
            tmp1.append(f"{In[k]}({Func.size_in(k)[0]}, {Func.size_in(k)[1]})")

        tmp2 = []
        for k in range(len(Out)):
            tmp2.append(f"{Out[k]}({Func.size_out(k)[0]}, {Func.size_out(k)[1]})")
        print(", ".join(tmp1) + " -> " + ", ".join(tmp2))
        print("\n")

        if Type == 'sfun':
            print(f"{dir}{filename}.c\n{dir}{filename}_Call.c\n\n")
            print(f'#include "{dir}{filename}.h"\n#include "{dir}{filename}_Call.h"\n\n')
            print(f'{filename}_Call_Func({", ".join(In)}, {", ".join(Out)});\n')

    #     z = f'#include "{filename}.h"\n#include "{filename}_Call.h"\n\n'
    #     # z += "#ifndef casadi_inf\n"
    #     # z += "\t#define casadi_inf 1e300\n"
    #     # z += "#endif\n\n"

    #     # z += "#ifndef casadi_nan\n"
    #     # z += "\t#define casadi_nan -1\n"
    #     # z += "#endif\n\n"

    #     Sfinal = 0
    #     for k in range(len(In)):
    #         Sfinal += Func.size_in(k)[0]*Func.size_in(k)[1]
    #     for k in range(len(Out)):
    #         Sfinal += Func.size_out(k)[0]*Func.size_out(k)[1]

    #     z += f'void {filename}_Call_Func({", ".join(["casadi_real *"+x for x in In])}, {", ".join(["casadi_real *"+x for x in Out])})\n'

    #     z += "{\n\tint i;\n"
    #     z += f"\tconst casadi_real *arg[{Func.sz_arg()}];\n"
    #     z += f"\tcasadi_real *res[{Func.sz_res()}];\n"
    #     z += f"\tcasadi_int iw[{Func.sz_iw()}];\n"
    #     z += f"\tcasadi_real w[{Func.sz_w() + Sfinal}];\n\n"

    #     S = 0
    #     for k in range(len(In)):
    #         z += f"\targ[{k}] = w + {S};\n"
    #         S += Func.size_in(k)[0]*Func.size_in(k)[1]
    #     z += "\n"
    #     for k in range(len(Out)):
    #         z += f"\tres[{k}] = w+{S};\n"
    #         S += Func.size_out(k)[0]*Func.size_out(k)[1]
    #     z += "\n"

    #     S = 0
    #     for k in range(len(In)):
    #         z += f"\tfor (i = {S}; i < {S+Func.size_in(k)[0]*Func.size_in(k)[1]}; i++)\n"
    #         z += f"\t\tw[i] = {In[k]}[i-{S}];\n\n"
    #         S += Func.size_in(k)[0]*Func.size_in(k)[1]
    #     z += "\n"

    #     z += f"\t{funcname}(arg, res, iw, w + {Sfinal}, 0);\n"
    #     z += "\n"

    #     for k in range(len(Out)):
    #         z += f"\tfor (i = {S}; i < {S+Func.size_out(k)[0]*Func.size_out(k)[1]}; i++)\n"
    #         z += f"\t\t{Out[k]}[i-{S}] = w[i];\n\n"
    #         S += Func.size_out(k)[0]*Func.size_out(k)[1]
    #     z += "\n"

    #     z += "}"

    #     with open(f"{dir}{filename}_Call.c", 'w') as fw:
    #         fw.write(z)

    #     z = f'#include "{filename}.h"\n\n'
    #     z += f'void {filename}_Call_Func({", ".join(["casadi_real *"+x for x in In])}, {", ".join(["casadi_real *"+x for x in Out])});\n'
    #     with open(f"{dir}{filename}_Call.h", 'w') as fw:
    #         fw.write(z)
    # else:
    #     print(f"Type {Type} not found!!!")
    #     return

    # tmp1 = []
    # for k in range(len(In)):
    #     tmp1.append(f"{In[k]}({Func.size_in(k)[0]}, {Func.size_in(k)[1]})")

    # tmp2 = []
    # for k in range(len(Out)):
    #     tmp2.append(f"{Out[k]}({Func.size_out(k)[0]}, {Func.size_out(k)[1]})")

    # if printhelp:
    #     print(", ".join(tmp1) + " -> " + ", ".join(tmp2))
    #     print("\n")

    #     if Type == 'sfun':
    #         print(f"{dir}{filename}.c\n{dir}{filename}_Call.c\n\n")
    #         print(f'#include "{dir}{filename}.h"\n#include "{dir}{filename}_Call.h"\n\n')
    #         print(f'{filename}_Call_Func({", ".join(In)}, {", ".join(Out)});\n')
