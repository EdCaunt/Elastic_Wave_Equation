# Modelling 2D VTI in Devito.
# Defined by elastic tensor components
# Centered source using Devito RickerSource object
# Will use spatially optimized scheme to propogate wavefield

from examples.seismic.source import RickerSource, TimeAxis
from devito import SpaceDimension, TimeDimension, Constant, Grid, TimeFunction, Function, NODE, Eq, Operator
from numpy import array, sqrt, rot90

so = 6 # Spatial derivatives are eighth order accurate
extent = (150., 150.)
# Grid is 2km x 2km with spacing 10m
shape = (151, 151)

dt = (1. / sqrt(2.)) / 2500. # Where did this come from again?
# Define x and z as spatial dimentions for Sympy
x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0]/(shape[0]-1)))
# Dimension called x, with constant spacing of 10 (called h_x) 
z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=extent[1]/(shape[1]-1)))
t = TimeDimension(name='t', spacing=Constant(name='dt', value=dt))
grid = Grid(extent=extent, shape=shape, dimensions=(x, z), time_dimension=t)

t0, tn = 0., 75. # Start and end times in ms
ns = int(tn/dt)
time_range = TimeAxis(start=t0, stop=tn, step=dt/2.) #Set up time axis object for ricker source

src = RickerSource(name='src', grid=grid, f0=0.100, time_range=time_range) # Ricker wavelet source
src.coordinates.data[:] = array([75., 75.])

rho = Function(name="rho", grid=grid, space_order=2) #Bouyancy
rho.data[:] = 1/1900.

# k_1,2,3,4 refer to components in the stress tensor as follows:
#
# | k_1 k_2  0  |
# |             |
# | k_2 k_3  0  |
# |             |
# |  0   0  k_4 |

# Adapted from Juhlin 1995 and Okaya and McEvilly 2003

# Tensor components can be spatially variant
k_1= Function(name="k1", grid=grid, space_order=2)
k_1.data[:] = 2700.*1.1 # Parameters scaled relative to values for isotropic case (lambda = 1300, mu = 700)
k_2= Function(name="k2", grid=grid, space_order=2)
k_2.data[:] = 1300.*1.
k_3= Function(name="k3", grid=grid, space_order=2)
k_3.data[:] = 2700.*1.
k_4= Function(name="k4", grid=grid, space_order=2)
k_4.data[:] = 700.*1.

# Now for the optimized scheme

# Grids are staggered to prevent pressure decoupling. Note that staggering is done in corresponding directions
vx_opt = TimeFunction(name='vx_opt', grid=grid, space_order=so) # Velocity field x
vz_opt = TimeFunction(name='vz_opt', grid=grid, space_order=so) # Velocity field z
# staggered=x entails discretization on x edges
txx_opt = TimeFunction(name='txx_opt', grid=grid, space_order=so)
tzz_opt = TimeFunction(name='tzz_opt', grid=grid, space_order=so)
txz_opt = TimeFunction(name='txz_opt', grid=grid, space_order=so)
# Stress axis for normal and shear stresses

# The source injection term
src_xx_opt = src.inject(field=txx_opt.forward, expr=src) 
src_zz_opt = src.inject(field=tzz_opt.forward, expr=src)

# Optimized stencil coefficients
#a_m3 = -0.02651995
#a_m2 = 0.18941314
#a_m1 = -0.79926643
#a_1 = 0.79926643
#a_2 = -0.18941314
#a_3 = 0.02651995

a_m3 = -1./60.
a_m2 = 3./20.
a_m1 = -3./4.
a_1 = 3./4.
a_2 = -3./20.
a_3 = 1./60.

h_x = grid.spacing[0]
h_z = grid.spacing[1]
time = grid.time_dim
x = grid.dimensions[0]
z = grid.dimensions[1]


# This doesn't work
#u_vx_opt = Eq(vx_opt.forward, vx_opt + dt*rho*((a_m3*txx_opt[time,x-3,z]
#                                                + a_m2*txx_opt[time,x-2,z]
#                                                + a_m1*txx_opt[time,x-1,z]
#                                                + a_1*txx_opt[time,x+1,z]
#                                                + a_2*txx_opt[time,x+2,z]
#                                                + a_3*txx_opt[time,x+3,z])/h_x + txz_opt.dz))

print(txx_opt.dx)

# Uncomment for working pde
u_vx_opt = Eq(vx_opt.forward, vx_opt + dt*rho*(txx_opt.dx + txz_opt.dz))
u_vz_opt = Eq(vz_opt.forward, vz_opt + dt*rho*(txz_opt.dx + tzz_opt.dz))
u_txx_opt = Eq(txx_opt.forward,
           txx_opt + dt*k_1*vx_opt.forward.dx 
           + dt*k_2*vz_opt.forward.dz)
u_tzz_opt = Eq(tzz_opt.forward,
           tzz_opt + dt*k_2*vx_opt.forward.dx 
           + dt*k_3*vz_opt.forward.dz)
u_txz_opt = Eq(txz_opt.forward,
          txz_opt + dt*k_4*(vx_opt.forward.dz + vz_opt.forward.dx))

op_opt = Operator([u_vx_opt, u_vz_opt, u_txx_opt, u_tzz_opt, u_txz_opt] + src_xx_opt + src_zz_opt)
#Source is injected in xx and zz directions

# Reset the fields
vx_opt.data[:] = 0. #Velocity components
vz_opt.data[:] = 0.
txx_opt.data[:] = 0. #Symmetric stress tensors
tzz_opt.data[:] = 0.
txz_opt.data[:] = 0.

op_opt.apply(t_M=ns)

import matplotlib.pyplot as plt
plt.imshow(vx_opt.data[0])
plt.show()