import matplotlib.pyplot as plt
from devito import TimeFunction, Grid, SpaceDimension, Operator, Constant, Eq
from sympy import solve
from numpy import sin, cos, pi, linspace, shape, clip
from scipy.integrate import quad

# Global constants
L = 10. # Define length of domain as a global variable
k = 100 # Number of terms in the Fourier sine series
l = 4001 # Define number of points in domain

extent = (L,) # Grid is L long with l grid points
shape = (l,)

x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0]/(shape[0]-1)))
grid = Grid(extent=extent, shape=shape, dimensions=(x,))

# Set up optimized FD parameters
so_opt = 4 # 6th order accurate in space
to_opt = 2 # 2nd order accurate in time
time = grid.time_dim
h_x = grid.spacing[0]


# dt is defined using Courant condition (c = 1)
dt = 0.5*(L/(shape[0]-1)) # Timestep is half critical dt (0.0025)
t_end = L # Standing wave will cycle twice(?) in time L
ns = int(t_end/dt) # Number of timesteps = total time/timestep size

# Set up function and stencil
u_opt = TimeFunction(name="u_dev", grid=grid, space_order=so_opt, time_order=to_opt, save=ns+1)

# Optimized stencil coefficients
a_0 = 5.71770701
a_1 = -4.82994692
a_2 = 2.38197877
a_3 = -0.41088535

stencil_opt = Eq(u_opt.forward, solve((a_3*u_opt[time, x - 3]
                                      + a_2*u_opt[time, x - 2]
                                      + a_1*u_opt[time, x - 1]
                                      + a_0*u_opt[time, x]
                                      + a_1*u_opt[time, x + 1]
                                      + a_2*u_opt[time, x + 2]
                                      + a_3*u_opt[time, x + 3])/h_x**2
                                      - u_opt.dt2, u_opt.forward)[0])

bc = [Eq(u_opt[time+1,0], 0.0)] # Specify boundary conditions

# Functions for initalizing standing square wave
def square_init(x): # Square function to base Fourier series off of
    if x >= 0 and x < L/4.:
        return 1.
    elif x >= L/4. and x < L/2.:
        return -1.
    else:
        return 0.

def b_n_inner(x, n): # Inner part of b_n to allow for scipy.integrate.quad to be used
    return square_init(x)*sin(2.*n*pi*x/L)

def b_n_calc(n):
    return (4./L)*quad(b_n_inner, 0, L/2., args=(n))[0]
    
def u_init(x):
    u_temp = 0.
    for n in range(1, k+1):
        u_temp += b_n_calc(n)*sin(2.*n*pi*x/L)
        
    return u_temp

# Initialize wavefield
u_opt.data[:] = u_init(linspace(0, L, u_opt.data.shape[1]))

# Create operator
op_opt = Operator([stencil_opt] + bc)

# Apply operator
op_opt.apply(time_M=ns-1, dt=dt)

x_vals = linspace(0, L, l)
for i in range(5):
    fig = plt.figure()
    plt.plot(x_vals, clip(u_opt.data[i], a_min=-1.2, a_max=1.2))
    plt.show()