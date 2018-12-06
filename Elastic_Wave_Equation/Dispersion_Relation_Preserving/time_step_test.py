import numpy as np
from devito import Grid, Function, TimeFunction, Eq, Operator, solve
from devito import ConditionalDimension

from devito import Constant, first_derivative, second_derivative
from devito import left, right
from examples.cfd import plot_field
import matplotlib.pyplot as plt
from matplotlib import cm

from numpy import exp

# define spatial mesh
# Size of rectangular domain
Lx = 1.
Ly = Lx

# Number of grid points in each direction, including boundary nodes
Nx = 11
Ny = Nx

# hence the mesh spacing
dx = Lx/(Nx-1)
dy = Ly/(Ny-1)

grid = Grid(shape=(Nx, Ny), extent=(Lx, Ly))
time = grid.time_dim
t = grid.stepping_dim
x, y = grid.dimensions

dt = 1e-4
t_end = 1.0
ns = int(t_end/dt)

# Set up three fields: They will all give the same solution but we will get there
# using three 'different' methods
u = TimeFunction(name='u', grid=grid)
u1 = TimeFunction(name='u1', grid=grid, save=ns+1) # Note the 'save=ns+1'
u2 = TimeFunction(name='u2', grid=grid, save=ns+1)

# Initialise
u.data[:] = 0.
u1.data[:] = 0.
u2.data[:] = 0.

# Initial conditions
u.data[0,:,:] = 1.
u1.data[0,:,:] = 1.
u2.data[0,:,:] = 1.

# Main equations
eq = Eq(u.dt, u)
stencil = solve(eq, u.forward)
eq1 = Eq(u1.dt, u1) # can re-use this equation (see below)
stencil1 = solve(eq1, u1.forward)

# Create the operators
op = Operator(Eq(u.forward, stencil))
op1 = Operator(Eq(u1.forward, stencil1))

# Method 1: function without 'save' specified
op.apply(time_m=0, time_M=ns-1, dt=dt)

# Method 2: function wit 'save' specified
op1.apply(time_m=0, time_M=ns-1, dt=dt)

# Method 3: Repeated calls (but avoid this since it's slow)
for j in range(0,ns):
    op1.apply(u1=u2, time_m=j, time_M=j, dt=dt)
    
# compare solutions at the end of the iteration
print(u.data[0,0,0], u1.data[-1,0,0], u2.data[-1,0,0], exp(1.))

# Note we can grab the value at any point in time from u1/u2 via:
# u1[10,x,y] etc. - this won't work for u since we didn't specify 'save=some_number'
