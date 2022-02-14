import casadi as ca
from pynlcontrol import QPInterface


x1 = ca.SX.sym('x1')
x2 = ca.SX.sym('x2')

a = ca.SX.sym('a')
b = ca.SX.sym('b')

J = (x1-a)**2 + (x2-b)**2
zOp = ca.SX.sym('zOp', 2)

H, g, _ = ca.quadratic_coeff(J, ca.vertcat(x1, x2))

A, _ = ca.linear_coeff(ca.vertcat(2*x1+3*x2, x1-x2), ca.vertcat(x1, x2))


qp = QPInterface.qpOASES(
    H, g, A=A, lbA=ca.vertcat(-ca.inf, 0), ubA=ca.vertcat(3, 0), p=[a, b])
qp.exportCode(
    'test1',
    dir="Test1_Exported",
    printsfun=True,
    mex=False,
    TestCode=False,
    Options={
        'max_iter': 5,
    }
)
