from devito import *
from examples.seismic.source import WaveletSource, RickerSource, GaborSource, TimeAxis
from examples.seismic import plot_image
import numpy as np
from numpy import swapaxes, amax
import matplotlib.pyplot as plt

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

time = grid.stepping_dim
#time = grid.time_dim
h_x = grid.spacing[0]
h_z = grid.spacing[1]

# Optimized stencil coefficients
# Type "A" (in stagger direction)
a_m2 = -0.00640097
a_m1 = 0.07367152
a_0 = -1.1890097
a_1 = 1.1890097
a_2 = -0.07367152
a_3 = 0.00640097
# Type "B" (not in stagger direction)
b_m3 = -0.00640097
b_m2 = 0.07367152
b_m1 = -1.1890097
b_0 = 1.1890097
b_1 = -0.07367152
b_2 = 0.00640097

#so = 4
so = 6
vx = TimeFunction(name='vx', grid=grid, staggered=x, space_order=so)
vz = TimeFunction(name='vz', grid=grid, staggered=z, space_order=so)
txx = TimeFunction(name='txx', grid=grid, staggered=NODE, space_order=so)
tzz = TimeFunction(name='tzz', grid=grid, staggered=NODE, space_order=so)
txz = TimeFunction(name='txz', grid=grid, staggered=(x, z), space_order=so)

vx_opt = TimeFunction(name='vx_opt', grid=grid, staggered=x, space_order=so)
vz_opt = TimeFunction(name='vz_opt', grid=grid, staggered=z, space_order=so)
txx_opt = TimeFunction(name='txx_opt', grid=grid, staggered=NODE, space_order=so)
tzz_opt = TimeFunction(name='tzz_opt', grid=grid, staggered=NODE, space_order=so)
txz_opt = TimeFunction(name='txz_opt', grid=grid, staggered=(x, z), space_order=so)


V_p = 4.0
V_s = 1.0
density = 3.

# The source injection term
src_xx = src.inject(field=txx.forward, expr=src)
src_zz = src.inject(field=tzz.forward, expr=src)

src_xx_opt = src.inject(field=txx_opt.forward, expr=src)
src_zz_opt = src.inject(field=tzz_opt.forward, expr=src)

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

# Optimized stencils
u_vx_opt = Eq(vx_opt.forward, vx_opt - dt*ro*((b_m3*txx_opt[time,x-3,z]
                                              + b_m2*txx_opt[time,x-2,z]
                                              + b_m1*txx_opt[time,x-1,z]
                                              + b_0*txx_opt[time,x,z]
                                              + b_1*txx_opt[time,x+1,z]
                                              + b_2*txx_opt[time,x+2,z])/h_x 
                                              + (a_m2*txz_opt[time,x,z-2]
                                                + a_m1*txz_opt[time,x,z-1]
                                                + a_0*txz_opt[time,x,z]
                                                + a_1*txz_opt[time,x,z+1]
                                                + a_2*txz_opt[time,x,z+2]
                                                + a_3*txz_opt[time,x,z+3])/h_z))

u_vz_opt = Eq(vz_opt.forward, vz_opt - ro*dt*((a_m2*txz_opt[time,x-2,z]
                                              + a_m1*txz_opt[time,x-1,z]
                                              + a_0*txz_opt[time,x,z]
                                              + a_1*txz_opt[time,x+1,z]
                                              + a_2*txz_opt[time,x+2,z]
                                              + a_3*txz_opt[time,x+3,z])/h_x 
                                              + (b_m3*tzz_opt[time,x,z-3]
                                                + b_m2*tzz_opt[time,x,z-2]
                                                + b_m1*tzz_opt[time,x,z-1]
                                                + b_0*tzz_opt[time,x,z]
                                                + b_1*tzz_opt[time,x,z+1]
                                                + b_2*tzz_opt[time,x,z+2])/h_z))

#u_vx_opt = Eq(vx_opt.forward, vx_opt - dt*ro*(txx_opt.dx + txz_opt.dz))

#u_vz_opt = Eq(vz_opt.forward, vz_opt - ro*dt*(txz_opt.dx + tzz_opt.dz))

#u_txx_opt = Eq(txx_opt.forward, txx_opt - (l+2*mu)*dt * vx_opt.forward.dx -  l*dt * vz_opt.forward.dz)

u_txx_opt = Eq(txx_opt.forward, txx_opt - (l+2*mu)*dt * (a_m2*vx_opt[time+1,x-2,z]
                                                        + a_m1*vx_opt[time+1,x-1,z]
                                                        + a_0*vx_opt[time+1,x,z]
                                                        + a_1*vx_opt[time+1,x+1,z]
                                                        + a_2*vx_opt[time+1,x+2,z]
                                                        + a_3*vx_opt[time+1,x+3,z])/h_x 
               - l*dt * (a_m2*vz_opt[time+1,x,z-2]
                        + a_m1*vz_opt[time+1,x,z-1]
                        + a_0*vz_opt[time+1,x,z]
                        + a_1*vz_opt[time+1,x,z+1]
                        + a_2*vz_opt[time+1,x,z+2]
                        + a_3*vz_opt[time+1,x,z+3])/h_z)

#u_tzz_opt = Eq(tzz_opt.forward, tzz_opt - (l+2*mu)*dt * vz_opt.forward.dz -  l*dt * vx_opt.forward.dx)

u_tzz_opt = Eq(tzz_opt.forward, tzz_opt - (l+2*mu)*dt * (a_m2*vz_opt[time+1,x,z-2]
                                                         + a_m1*vz_opt[time+1,x,z-1]
                                                         + a_0*vz_opt[time+1,x,z]
                                                         + a_1*vz_opt[time+1,x,z+1]
                                                         + a_2*vz_opt[time+1,x,z+2]
                                                         + a_3*vz_opt[time+1,x,z+3])/h_z 
               - l*dt * (a_m2*vx_opt[time+1,x-2,z]
                         + a_m1*vx_opt[time+1,x-1,z]
                         + a_0*vx_opt[time+1,x,z]
                         + a_1*vx_opt[time+1,x+1,z]
                         + a_2*vx_opt[time+1,x+2,z]
                         + a_3*vx_opt[time+1,x+3,z])/h_x)

#u_txz_opt = Eq(txz_opt.forward, txz_opt - mu*dt * (vx_opt.forward.dz + vz_opt.forward.dx))

u_txz_opt = Eq(txz_opt.forward, txz_opt - mu*dt * ((b_m3*vx_opt[time+1,x,z-3]
                                                   + b_m2*vx_opt[time+1,x,z-2]
                                                   + b_m1*vx_opt[time+1,x,z-1]
                                                   + b_0*vx_opt[time+1,x,z]
                                                   + b_1*vx_opt[time+1,x,z+1]
                                                   + b_2*vx_opt[time+1,x,z+2])/h_z 
                                                   + (b_m3*vz_opt[time+1,x-3,z]
                                                     + b_m2*vz_opt[time+1,x-2,z]
                                                     + b_m1*vz_opt[time+1,x-1,z]
                                                     + b_0*vz_opt[time+1,x,z]
                                                     + b_1*vz_opt[time+1,x+1,z]
                                                     + b_2*vz_opt[time+1,x+2,z])/h_x))



op = Operator([u_vx, u_vz, u_txx, u_tzz, u_txz] + src_xx + src_zz)

op_opt = Operator([u_vx_opt, u_vz_opt, u_txx_opt, u_tzz_opt, u_txz_opt] + src_xx_opt + src_zz_opt)
# Might need to add bcs

# Reset the fields
vx.data[:] = 0.
vz.data[:] = 0.
txx.data[:] = 0.
tzz.data[:] = 0.
txz.data[:] = 0.

op()

# Reset the fields
vx_opt.data[:] = 0.
vz_opt.data[:] = 0.
txx_opt.data[:] = 0.
tzz_opt.data[:] = 0.
txz_opt.data[:] = 0.

op_opt()

plt.imshow(swapaxes(vx_opt.data[1], 0, 1), interpolation='bicubic', extent=[0,2000,2000,0], vmin = -0.025, vmax = 0.025)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title('$v_x$ at t = %.2f ms; Optimized 4th order\nCourant number = %.2f' % (tn, 4.*dt/10.))
plt.colorbar(label="Velocity ($ms^{-1}$)")
plt.savefig("Figures/2D_Elastic/%.0f_vx_opt" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(vz_opt.data[1], 0, 1), interpolation='bicubic', extent=[0,2000,2000,0], vmin = -0.025, vmax = 0.025)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title('$v_z$ at t = %.2f ms; Optimized 4th order\nCourant number = %.2f' % (tn, 4.*dt/10.))
plt.colorbar(label="Velocity ($ms^{-1}$)")
plt.savefig("Figures/2D_Elastic/%.0f_vz_opt" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(txx_opt.data[1], 0, 1), interpolation='bicubic', extent=[0,2000,2000,0], vmin = -0.15, vmax = 0.15)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title('${\\tau}_{xx}$ at t = %.2f ms; Optimized 4th order\nCourant number = %.2f' % (tn, 4.*dt/10.))
plt.colorbar(label="Stress (Pa)")
plt.savefig("Figures/2D_Elastic/%.0f_txx_opt" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(tzz_opt.data[1], 0, 1), interpolation='bicubic', extent=[0,2000,2000,0], vmin = -0.15, vmax = 0.15)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title('${\\tau}_{zz}$ at t = %.2f ms; Optimized 4th order\nCourant number = %.2f' % (tn, 4.*dt/10.))
plt.colorbar(label="Stress (Pa)")
plt.savefig("Figures/2D_Elastic/%.0f_tzz_opt" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(txz_opt.data[1], 0, 1), interpolation='bicubic', extent=[0,2000,2000,0], vmin = -0.01, vmax = 0.01)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title('${\\tau}_{xz}$ at t = %.2f ms; Optimized 4th order\nCourant number = %.2f' % (tn, 4.*dt/10.))
plt.colorbar(label="Stress (Pa)")
plt.savefig("Figures/2D_Elastic/%.0f_txz_opt" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(vx.data[1], 0, 1), interpolation='bicubic', extent=[0,2000,2000,0], vmin = -0.025, vmax = 0.025)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title('$v_x$ at t = %.2f ms; Conventional 6th order\nCourant number = %.2f' % (tn, 4.*dt/10.))
plt.colorbar(label="Velocity ($ms^{-1}$)")
plt.savefig("Figures/2D_Elastic/%.0f_vx_dev" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(vz.data[1], 0, 1), interpolation='bicubic', extent=[0,2000,2000,0], vmin = -0.025, vmax = 0.025)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title('$v_z$ at t = %.2f ms; Conventional 6th order\nCourant number = %.2f' % (tn, 4.*dt/10.))
plt.colorbar(label="Velocity ($ms^{-1}$)")
plt.savefig("Figures/2D_Elastic/%.0f_vz_dev" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(txx.data[1], 0, 1), interpolation='bicubic', extent=[0,2000,2000,0], vmin = -0.15, vmax = 0.15)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title('${\\tau}_{xx}$ at t = %.2f ms; Conventional 6th order\nCourant number = %.2f' % (tn, 4.*dt/10.))
plt.colorbar(label="Stress (Pa)")
plt.savefig("Figures/2D_Elastic/%.0f_txx_dev" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(tzz.data[1], 0, 1), interpolation='bicubic', extent=[0,2000,2000,0], vmin = -0.15, vmax = 0.15)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title('${\\tau}_{zz}$ at t = %.2f ms; Conventional 6th order\nCourant number = %.2f' % (tn, 4.*dt/10.))
plt.colorbar(label="Stress (Pa)")
plt.savefig("Figures/2D_Elastic/%.0f_tzz_dev" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(txz.data[1], 0, 1), interpolation='bicubic', extent=[0,2000,2000,0], vmin = -0.01, vmax = 0.01)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title('${\\tau}_{xz}$ at t = %.2f ms; Conventional 6th order\nCourant number = %.2f' % (tn, 4.*dt/10.))
plt.colorbar(label="Stress (Pa)")
plt.savefig("Figures/2D_Elastic/%.0f_txz_dev" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(100.*(vx_opt.data[1]-vx.data[1])/amax(abs(vx.data[1])), 0, 1), 
           interpolation='bicubic', extent=[0,2000,2000,0], vmin = -2., vmax = 2.)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title('$v_x$ at t = %.2f ms; Percentage difference\nCourant number = %.2f' % (tn, 4.*dt/10.))
plt.colorbar(label="Difference (% of maximum)")
plt.savefig("Figures/2D_Elastic/%.0f_vx_dif" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(100.*(vz_opt.data[1]-vz.data[1])/amax(abs(vz.data[1])), 0, 1), 
           interpolation='bicubic', extent=[0,2000,2000,0], vmin = -2., vmax = 2.)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title('$v_z$ at t = %.2f ms; Percentage difference\nCourant number = %.2f' % (tn, 4.*dt/10.))
plt.colorbar(label="Difference (% of maximum)")
plt.savefig("Figures/2D_Elastic/%.0f_vz_dif" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(100.*(txx_opt.data[1]-txx.data[1])/amax(abs(txx.data[1])), 0, 1), 
           interpolation='bicubic', extent=[0,2000,2000,0], vmin = -2., vmax = 2.)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title('${\\tau}_{xx}$ at t = %.2f ms; Percentage difference\nCourant number = %.2f' % (tn, 4.*dt/10.))
plt.colorbar(label="Difference (% of maximum)")
plt.savefig("Figures/2D_Elastic/%.0f_txx_dif" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(100.*(tzz_opt.data[1]-tzz.data[1])/amax(abs(tzz.data[1])), 0, 1), 
           interpolation='bicubic', extent=[0,2000,2000,0], vmin = -2., vmax = 2.)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title('${\\tau}_{zz}$ at t = %.2f ms; Percentage difference\nCourant number = %.2f' % (tn, 4.*dt/10.))
plt.colorbar(label="Difference (% of maximum)")
plt.savefig("Figures/2D_Elastic/%.0f_tzz_dif" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(100.*(txz_opt.data[1]-txz.data[1])/amax(abs(txz.data[1])), 0, 1), 
           interpolation='bicubic', extent=[0,2000,2000,0], vmin = -2., vmax = 2.)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title('${\\tau}_{xz}$ at t = %.2f ms; Percentage difference\nCourant number = %.2f' % (tn, 4.*dt/10.))
plt.colorbar(label="Difference (% of maximum)")
plt.savefig("Figures/2D_Elastic/%.0f_txz_dif" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(vx_opt.data[1]-vx.data[1], 0, 1), 
           interpolation='bicubic', extent=[0,2000,2000,0], vmin = -0.025, vmax = 0.025)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title('$v_x$ at t = %.2f ms; Absolute difference\nCourant number = %.2f' % (tn, 4.*dt/10.))
plt.colorbar(label="Velocity ($ms^{-1}$)")
plt.savefig("Figures/2D_Elastic/%.0f_vx_abs" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(vz_opt.data[1]-vz.data[1], 0, 1), 
           interpolation='bicubic', extent=[0,2000,2000,0], vmin = -0.025, vmax = 0.025)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title('$v_z$ at t = %.2f ms; Absolute difference\nCourant number = %.2f' % (tn, 4.*dt/10.))
plt.colorbar(label="Velocity ($ms^{-1}$)")
plt.savefig("Figures/2D_Elastic/%.0f_vz_abs" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(txx_opt.data[1]-txx.data[1], 0, 1), 
           interpolation='bicubic', extent=[0,2000,2000,0], vmin = -0.15, vmax = 0.15)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title('${\\tau}_{xx}$ at t = %.2f ms; Absolute difference\nCourant number = %.2f' % (tn, 4.*dt/10.))
plt.colorbar(label="Stress (Pa)")
plt.savefig("Figures/2D_Elastic/%.0f_txx_abs" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(tzz_opt.data[1]-tzz.data[1], 0, 1), 
           interpolation='bicubic', extent=[0,2000,2000,0], vmin = -0.15, vmax = 0.15)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title('${\\tau}_{zz}$ at t = %.2f ms; Absolute difference\nCourant number = %.2f' % (tn, 4.*dt/10.))
plt.colorbar(label="Stress (Pa)")
plt.savefig("Figures/2D_Elastic/%.0f_tzz_abs" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(txz_opt.data[1]-txz.data[1], 0, 1), 
           interpolation='bicubic', extent=[0,2000,2000,0], vmin = -0.01, vmax = 0.01)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title('${\\tau}_{xz}$ at t = %.2f ms; Absolute difference\nCourant number = %.2f' % (tn, 4.*dt/10.))
plt.colorbar(label="Stress (Pa)")
plt.savefig("Figures/2D_Elastic/%.0f_txz_abs" % (tn), dpi=200)
plt.show()