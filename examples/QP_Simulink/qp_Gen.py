import casadi as ca
from pynlcontrol import QPInterface


x1 = ca.SX.sym('x1')
x2 = ca.SX.sym('x2')

a = ca.SX.sym('a')
b = ca.SX.sym('b')

J = (x1-a)**2 + (x2-b)**2

H, h, _ = ca.quadratic_coeff(J, ca.vertcat(x1, x2))

g = ca.vertcat(2*x1+3*x2, x1-x2), ca.vertcat(x1, x2)
lbg = ca.vertcat(-ca.inf, 0)
ubg = ca.vertcat(3, 10)

A, c = ca.linear_coeff(ca.vertcat(2*x1+3*x2, x1-x2), ca.vertcat(x1, x2))
lbA = lbg - c
ubA = ubg - c


qp = QPInterface.qpOASES(H, h, A=A, lbA=lbA, ubA=ubA, p=[a, b])
qp.exportCode('test1', dir='Test1_Exported', printsfun=True, mex=False, TestCode=False, Options={'max_iter': 5})
