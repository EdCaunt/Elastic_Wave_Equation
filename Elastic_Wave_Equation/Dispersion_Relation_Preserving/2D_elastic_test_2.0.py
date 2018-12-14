from devito import *
from examples.seismic.source import WaveletSource, RickerSource, GaborSource, TimeAxis
from examples.seismic import plot_image
import numpy as np

from sympy import init_printing, latex
init_printing(use_latex=True)

# Initial grid: 1km x 1km, with spacing 100m
extent = (2000., 2000.)
#Fairly sure grid is actually 2km x 2km with spacing 10m
shape = (201, 201)
#Define x and z as spatial dimentions for Sympy
x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0]/(shape[0]-1)))
#Dimension called x, with constant spacing of 10 (called h_x) 
z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=extent[1]/(shape[1]-1)))
grid = Grid(extent=extent, shape=shape, dimensions=(x, z))

# Timestep size from Eq. 7 with V_p=6000. and dx=100
t0, tn = 0., 600.
dt = (10. / np.sqrt(2.)) / 6.
time_range = TimeAxis(start=t0, stop=tn, step=dt)

src = RickerSource(name='src', grid=grid, f0=0.01, time_range=time_range)
src.coordinates.data[:] = np.array([1000., 1000.])
src.show()

# Now that looks pretty! But let's do it again with a higher order...
so = 6
vx= TimeFunction(name='vx', grid=grid, staggered=x, space_order=so)
vz = TimeFunction(name='vz', grid=grid, staggered=z, space_order=so)
txx = TimeFunction(name='txx', grid=grid, staggered=NODE, space_order=so)
tzz = TimeFunction(name='tzz', grid=grid, staggered=NODE, space_order=so)
txz = TimeFunction(name='txz', grid=grid, staggered=(x, z), space_order=so)

print("vx.dx stencil", vx.dx, "\n\n")
print("vx.dz stencil", vx.dz, "\n\n")
print("vz.dx stencil", vz.dx, "\n\n")
print("vz.dz stencil", vz.dz, "\n\n")
print("txx.dx stencil", txx.dx, "\n\n")
print("txx.dz stencil", txx.dz, "\n\n")
print("tzz.dx stencil", tzz.dx, "\n\n")
print("tzz.dz stencil", tzz.dz, "\n\n")
print("txz.dx stencil", txz.dx, "\n\n")
print("txz.dz stencil", txz.dz, "\n\n")


V_p = 4.0
V_s = 1.0
density = 3.

# The source injection term
src_xx = src.inject(field=txx.forward, expr=src)
src_zz = src.inject(field=tzz.forward, expr=src)

#c1 = 9.0/8.0;
#c2 = -1.0/24.0;

# Thorbecke's parameter notation
cp2 = V_p*V_p
cs2 = V_s*V_s
ro = 1/density

mu = cs2*ro
l = (cp2*ro - 2*mu)

# fdelmodc reference implementation
u_vx = Eq(vx.forward, vx - dt*ro*(txx.dx + txz.dz))

u_vz = Eq(vz.forward, vz - ro*dt*(txz.dx + tzz.dz))

u_txx = Eq(txx.forward, txx - (l+2*mu)*dt * vx.forward.dx -  l*dt * vz.forward.dz)
u_tzz = Eq(tzz.forward, tzz - (l+2*mu)*dt * vz.forward.dz -  l*dt * vx.forward.dx)

u_txz = Eq(txz.forward, txz - mu*dt * (vx.forward.dz + vz.forward.dx))

op = Operator([u_vx, u_vz, u_txx, u_tzz, u_txz] + src_xx + src_zz)

# Reset the fields
vx.data[:] = 0.
vz.data[:] = 0.
txx.data[:] = 0.
tzz.data[:] = 0.
txz.data[:] = 0.

op()

plot_image(vx.data[0], vmin=-.5*1e-2, vmax=.5*1e-2, cmap="seismic")
plot_image(vz.data[0], vmin=-.5*1e-2, vmax=.5*1e-2, cmap="seismic")
plot_image(txx.data[0], vmin=-.5*1e-2, vmax=.5*1e-2, cmap="seismic")
plot_image(tzz.data[0], vmin=-.5*1e-2, vmax=.5*1e-2, cmap="seismic")
plot_image(txz.data[0], vmin=-.5*1e-2, vmax=.5*1e-2, cmap="seismic")