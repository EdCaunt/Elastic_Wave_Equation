#from devito import SpaceDimension, Constant, Grid, TimeFunction, Eq, Operator
#from sympy import solve
from numpy import sin, pi, linspace, shape
#import matplotlib.pyplot as plt

import numpy as np
from devito import Grid, Function, TimeFunction, Eq, Operator, solve
from devito import ConditionalDimension, Constant, first_derivative, second_derivative
from devito import left, right
from math import exp

import matplotlib
import matplotlib.pyplot as plt

L = 10. # Length of domain 
l = 3001 # Number of points in domain
# This behaviour can be replicated for pretty much any dx
x_vals = linspace(0, L, l)

so_dev = 4 # 4th order accurate in space
to_dev = 2 # 2nd order accurate in time

extent = (L,) # Grid is L long with l grid points
shape = (l,)


#dt = 1.8e-2 # blows up (courant number = 0.9)
#dt = 1e-2 # works
#dt = 1e-3 # works
#dt = 1e-4 # looks really wonky and amplitude increases
#dt = 1e-5 # very unstable, blows up
# why does an adequately small timestep lead to instability?

dt = 0.2*L/(l-1)

t_end = L # Standing wave will cycle twice in time L
ns = int(t_end/dt) # Number of timesteps = total time/timestep size

# Define x as spatial dimention for Sympy
#x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0]/(shape[0]-1)))
#grid = Grid(extent=extent, shape=shape, dimensions=(x,))
grid = Grid(shape=(l), extent=(L))
time = grid.time_dim

# Set up function and stencil
u_dev = TimeFunction(name="u_dev", grid=grid, space_order=so_dev, time_order=to_dev, save=ns+1)

u_dev.data[:] = 0.1*sin(2*pi*x_vals/L)

eq = Eq(u_dev.dt2-u_dev.dx2)
#stencil_dev = Eq(u_dev.forward, solve(u_dev.dx2 - u_dev.dt2, u_dev.forward)[0])
stencil_dev = solve(eq, u_dev.forward)

#print(stencil_dev)

bc = [Eq(u_dev[time+1,0], 0.0)] # Specify boundary conditions
bc += [Eq(u_dev[time+1,-1], 0.0)]
#print(bc)



fig = plt.figure()
plt.title("Initial wavefield")
plt.plot(x_vals, u_dev.data[1])
plt.show()

# Create operator
#op_dev = Operator([stencil_dev] + bc)
op_dev = Operator([Eq(u_dev.forward, stencil_dev)]+bc)
# Apply operator
op_dev.apply(time_M=ns-1, dt=dt)

fig = plt.figure()
plt.title("Wavefield after 2 cycles")
plt.plot(x_vals, u_dev.data[-1])
plt.show()

print(max(abs(0.1*sin(2*pi*x_vals/L)-u_dev.data[-1])))

fig = plt.figure()
plt.title("Difference")
plt.plot(x_vals, 0.1*sin(2*pi*x_vals/L) - u_dev.data[-1])
plt.show()