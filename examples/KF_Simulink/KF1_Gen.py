from pynlcontrol import Estimator, BasicUtils
import casadi as ca

Q11 = ca.SX.sym('Q11')
Q22 = ca.SX.sym('Q22')
Q33 = ca.SX.sym('Q33')
Q = BasicUtils.directSum([Q11, Q22, Q33])

R11 = ca.SX.sym('R11')
R22 = ca.SX.sym('R22')
R = BasicUtils.directSum([R11, R22])

A = ca.SX([[-0.4,0.1,-2],[0,-0.3,4],[1,0,0]])
B = ca.SX([[1,1],[0,1],[1,0]])
C = ca.SX([[1, 0, 0], [0, 1, 0]])
D = ca.SX([[0, 0], [0, 0]])

In, Out, InName, OutName = Estimator.KF(A, B, C, D, Q, R, 0.1)

KF_func = ca.Function('KF_func', In + [Q11, Q22, Q33, R11, R22], Out, InName + ['Q11', 'Q22', 'Q33', 'R11', 'R22'], OutName)

BasicUtils.Gen_Code(KF_func, 'KF_code', printhelp=True)

