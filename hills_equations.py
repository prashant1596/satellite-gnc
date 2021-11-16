import sympy as sym
from sympy.functions.elementary.complexes import Abs
from absolute_attitude_dynamics import skew
from sympy.matrices import Matrix

# Forces input
Fx, Fy, Fz = \
    sym.symbols('Fx Fy Fz', real=True)

# Chaser mass and gravitational params
m, mu = \
    sym.symbols('m mu', real=True)

# Target params
rt, w0 = \
    sym.symbols('rt w0', real=True)

# Relative position
sx, sy, sz = \
    sym.symbols('sx, sy, sz', real=True)

# Relative Velocity
dsx, dsy, dsz = \
    sym.symbols('dsx dsy dsz', real=True)

# Vectors in orbital frame
r = Matrix([0, 0, -rt])
w = Matrix([0, -w0, 0])
s = Matrix([sx, sy, sz])
ds = Matrix([dsx, dsy, dsz])
F = Matrix([Fx, Fy, Fz])
sw = skew(w)
spr = (s+r).T*(s+r)
norm_spr = (sym.sqrt(spr[0]))**3
acc = (sw*sw)*s \
        - (2*sw*ds) \
        + (w0**2)*r \
        - (mu*(s+r)/norm_spr) \
        + (F/m)

# Calculating jacobians
concat_matrix = ds.col_join(acc)
concat_var = s.col_join(ds)
A = concat_matrix.jacobian(concat_var)
B = concat_matrix.jacobian(F)

# Linearion points
val_subs = {
    sx : 0,
    sy : 0,
    sz : 0,
    dsx : 0,
    dsy : 0,
    dsz : 0,
    Fx : 0,
    Fy : 0,
    Fz : 0
}

A = A.subs(val_subs)
B = B.subs(val_subs)
A = A.subs(((mu*Abs(rt))/(Abs(rt)**4)), w0**2)
A = A.subs((mu/(Abs(rt)**3)), w0**2)
