from sympy import Symbol, linsolve, pretty, simplify
from sympy import S
from sympy.calculus import finite_diff_weights

import numpy as np

#Wm3 = Symbol("Wm3")
#Wm2 = Symbol("Wm2")
#Wm1 = Symbol("Wm1")
W0 = Symbol("W0")
W1 = Symbol("W1")
W2 = Symbol("W2")
W3 = Symbol("W3")
#Wb = Symbol("Wb")

#eta = Symbol("eta")

#eq1 = Wm2+Wm1+W0+W1+W2
#eq2 = -2*Wm2-Wm1+W1+2*W2
#eq3 = 2*Wm2+1/2*Wm1+1/2*W1+2*W2-1
#eq4 = -8/6*Wm2-1/6*Wm1+1/6*W1+8/6*W2
#eq5 = 16/24*Wm2+1/24*Wm1+1/24*W1+16/24*W2

eq1 = W0 + 2*W1 + 2*W2 + 2*W3
eq2 = W1 + 4*W2 + 9*W3 -1
eq3 = (1/12)*W1 + (4/3)*W2 + (27/4)*W3

Ws = linsolve([eq1, eq2, eq3], (W0, W1, W2, W3))

print(Ws)