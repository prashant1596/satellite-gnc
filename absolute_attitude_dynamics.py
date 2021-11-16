import numpy as np
import pandas as pd
import sympy as sym
from sympy.matrices import Matrix
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.physics.vector import ReferenceFrame
import math

def skew(vec):
    ret = Matrix([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])
    return ret

# Euler angles and derivatives
alpha, beta, gamma, dalpha, dbeta, dgamma = \
    sym.symbols('alpha beta gamma dalpha dbeta dgamma',
    real = True)

# Rotation rates and their derivatives
wx, wy, wz, dwx, dwy, dwz, w0 = \
    sym.symbols('wx wy wz dwx dwy dwz w0',
    real = True)

# Torque input
Tx, Ty, Tz = \
    sym.symbols('Tx Ty Tz',
    real = True)

# Inertia tensor
I11, I12, I13, I21, I22, I23, I31, I32, I33 = \
    sym.symbols('I11 I12 I13 I21 I22 I23 I31 I32 I33',
    real = True)

# Linearization points for the attitude
a, b, c = \
    sym.symbols('a b c',
    real = True)


# Kineamatics
angle = [alpha, beta, gamma]

# Angular velocity
omega = Matrix([wx, wy, wz])
omega0 = Matrix([0, -w0, 0])

cg = sym.cos(gamma)
sg = sym.sin(gamma)
cb = sym.cos(beta)
sb = sym.sin(beta)
ca = sym.cos(alpha)
sa = sym.sin(alpha)

B_angle = 1/cb * \
    Matrix([[cg, -sg, 0],[(cb*sg), (cb*cg), 0],[(-sb*cg), (sb*sg), cb]])
d_angle = B_angle * omega

# Dynamics
I = Matrix([[I11, I12, I13], [I21, I22, I23], [I31, I32, I33]])
I = Matrix([[I11, 0, 0], [0, I22, 0], [0, 0, I33]])
T = Matrix([Tx, Ty, Tz])

R3_g = Matrix([[cg, sg, 0], [-sg, cg, 0], [0, 0, 1]])
R2_b = Matrix([[cb, 0, -sb], [0, 1, 0], [sb, 0, cb]])
R1_a = Matrix([[1, 0 ,0], [0, ca, sa], [0, -sa, ca]])

A_bo = R3_g * R2_b * R1_a
w = omega + A_bo*omega0

sw = skew(w)

d_omega = I.inv()*(T-(sw*(I*w)))
dwx = d_omega[0]
dwy = d_omega[1]
dwz = d_omega[2]

# Jacobian computations
# Kineamatics Jacobian
f1 = d_angle
F1 = f1.jacobian(angle)
temp = f1.jacobian(omega)
F1 = F1.row_join(temp)
B1 = f1.jacobian(T)

# Dynamics Jacobian
f2 = d_omega
F2 = f2.jacobian(angle)
temp = f2.jacobian(omega)
F2 = F2.row_join(temp)
B2 = f2.jacobian(T)

# Linearization points
angle_vals = {alpha : 0,
              beta : 0,
              gamma : 0,
              wx : 0,
              wy : 0,
              wz : 0}

torque_vals = {Tx : 0,
               Ty : 0,
               Tz : 0}

# Evaluate Jacobian to build state-space matrices
f11 = f1.subs(angle_vals)
F11 = F1.subs(angle_vals)
F11 = F11.subs((sym.sin(b)/sym.cos(b)), sym.tan(b))
B11 = B1.subs(angle_vals)

subs_vals = dict(angle_vals)
subs_vals.update(torque_vals)
f21 = f2.subs(subs_vals)
F21 = F2.subs(subs_vals)
B21 = B2.subs(subs_vals)

A = F11.col_join(F21)
B = B11.col_join(B21)

A = sym.simplify(A)
B = sym.simplify(B)
