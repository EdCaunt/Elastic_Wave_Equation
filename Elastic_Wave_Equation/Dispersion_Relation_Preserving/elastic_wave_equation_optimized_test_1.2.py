from devito import Grid, Function, TimeFunction, NODE, Eq, Operator
from examples.seismic.source import RickerSource, TimeAxis
from numpy import sqrt, swapaxes
import matplotlib.pyplot as plt

extent = (150., 150.)
shape = (301, 301)

grid = Grid(extent=extent, shape=shape)

t0, tn = 0., 5. # Strange tiling effect at tn > 30
dt = (extent[0]/float(shape[0]-1))*(1. / sqrt(2.)) / 2500.
time_range = TimeAxis(start=t0, stop=tn, step=dt) #Set up time axis object for ricker source

src = RickerSource(name='src', grid=grid, f0=0.175, time_range=time_range) # Ricker wavelet source
src.coordinates.data[:] = [75., 75.] 

so = 6 # Spatial derivatives are second order accurate

x = grid.dimensions[0]
y = grid.dimensions[1]

vx_dev = TimeFunction(name='vx_dev', grid=grid, space_order=so) 
vy_dev = TimeFunction(name='vy_dev', grid=grid, space_order=so) 

txx_dev = TimeFunction(name='txx_dev', grid=grid, space_order=so)
tyy_dev = TimeFunction(name='tyy_dev', grid=grid, space_order=so)
txy_dev = TimeFunction(name='txy_dev', grid=grid, space_order=so)

vx = TimeFunction(name='vx', grid=grid, space_order=so) 
vy = TimeFunction(name='vy', grid=grid, space_order=so) 

txx = TimeFunction(name='txx', grid=grid, space_order=so)
tyy = TimeFunction(name='tyy', grid=grid, space_order=so)
txy = TimeFunction(name='txy', grid=grid, space_order=so)

V_p = Function(name="V_p", grid=grid, so=2)
V_p.data[:, 150:] = 2500. # Lower
V_p.data[:, :150] = 2000. # Upper
V_s = Function(name="V_s", grid=grid, so=2)
V_s.data[:, 150:] = 1500.
V_s.data[:, :150] = 1000.
rho = Function(name="rho", grid=grid, so=2)
rho.data[:, 150:] = 1/1900.
rho.data[:, :150] = 1/1500.

src_xx = src.inject(field=txx.forward, expr=src) 
src_yy = src.inject(field=tyy.forward, expr=src)

src_xx_dev = src.inject(field=txx_dev.forward, expr=src) 
src_yy_dev = src.inject(field=tyy_dev.forward, expr=src)

# Optimized stencil coefficients
a_m3 = -0.02651995
a_m2 = 0.18941314
a_m1 = -0.79926643
a_1 = 0.79926643
a_2 = -0.18941314
a_3 = 0.02651995

dx = grid.spacing[0]
dy = grid.spacing[1]
time = grid.stepping_dim

bc_vx = [Eq(vx[time+1,0,y], 0.0)]
bc_vx += [Eq(vx[time+1,300,y], 0.0)]
bc_vx += [Eq(vx[time+1,x,0], 0.0)]
bc_vx += [Eq(vx[time+1,x,300], 0.0)]

u_vx = Eq(vx.forward, vx + dt*rho*((a_m3*txx[time,x-3,y] 
                                   + a_m2*txx[time,x-2,y]
                                   + a_m1*txx[time,x-1,y]
                                   + a_1*txx[time,x+1,y]
                                   + a_2*txx[time,x+2,y]
                                   + a_3*txx[time,x+3,y])/dx 
                                   + (a_m3*txy[time,x,y-3]
                                     + a_m2*txy[time,x,y-2]
                                     + a_m1*txy[time,x,y-1]
                                     + a_1*txy[time,x,y+1]
                                     + a_2*txy[time,x,y+2]
                                     + a_3*txy[time,x,y+3])/dy))

u_vx_dev = Eq(vx_dev.forward, vx_dev + dt*rho*(txx_dev.dx + txy_dev.dy))

bc_vy = [Eq(vy[time+1,0,y], 0.0)]
bc_vy += [Eq(vy[time+1,300,y], 0.0)]
bc_vy += [Eq(vy[time+1,x,0], 0.0)]
bc_vy += [Eq(vy[time+1,x,300], 0.0)]

u_vy = Eq(vy.forward, vy + dt*rho*((a_m3*txy[time,x-3,y] 
                                   + a_m2*txy[time,x-2,y]
                                   + a_m1*txy[time,x-1,y]
                                   + a_1*txy[time,x+1,y]
                                   + a_2*txy[time,x+2,y]
                                   + a_3*txy[time,x+3,y])/dx 
                                   + (a_m3*tyy[time,x,y-3]
                                     + a_m2*tyy[time,x,y-2]
                                     + a_m1*tyy[time,x,y-1]
                                     + a_1*tyy[time,x,y+1]
                                     + a_2*tyy[time,x,y+2]
                                     + a_3*tyy[time,x,y+3])/dy))

u_vy_dev = Eq(vy_dev.forward, vy_dev + dt*rho*(txy_dev.dx + tyy_dev.dy))

bc_txx = [Eq(txx[time+1,0,y], 0.0)]
bc_txx += [Eq(txx[time+1,300,y], 0.0)]
bc_txx += [Eq(txx[time+1,x,0], 0.0)]
bc_txx += [Eq(txx[time+1,x,300], 0.0)]

u_txx = Eq(txx.forward, txx + (rho*V_p**2)*dt * (a_m3*vx[time+1,x-3,y] 
                                                 + a_m2*vx[time+1,x-2,y]
                                                 + a_m1*vx[time+1,x-1,y]
                                                 + a_1*vx[time+1,x+1,y]
                                                 + a_2*vx[time+1,x+2,y]
                                                 + a_3*vx[time+1,x+3,y])/dx 
           + (rho*V_p**2-2*rho*V_s**2)*dt * (a_m3*vy[time+1,x,y-3]
                                             + a_m2*vy[time+1,x,y-2]
                                             + a_m1*vy[time+1,x,y-1]
                                             + a_1*vy[time+1,x,y+1]
                                             + a_2*vy[time+1,x,y+2]
                                             + a_3*vy[time+1,x,y+3])/dy)


u_txx_dev = Eq(txx_dev.forward, txx_dev + (rho*V_p**2)*dt * vx_dev.forward.dx + (rho*V_p**2-2*rho*V_s**2)*dt * vy_dev.forward.dy)

bc_tyy = [Eq(tyy[time+1,0,y], 0.0)]
bc_tyy += [Eq(tyy[time+1,300,y], 0.0)]
bc_tyy += [Eq(tyy[time+1,x,0], 0.0)]
bc_tyy += [Eq(tyy[time+1,x,300], 0.0)]

u_tyy = Eq(tyy.forward, tyy + (rho*V_p**2)*dt * (a_m3*vy[time+1,x,y-3]
                                                 + a_m2*vy[time+1,x,y-2]
                                                 + a_m1*vy[time+1,x,y-1]
                                                 + a_1*vy[time+1,x,y+1]
                                                 + a_2*vy[time+1,x,y+2]
                                                 + a_3*vy[time+1,x,y+3])/dy
           + (rho*V_p**2-2*rho*V_s**2)*dt * (a_m3*vx[time+1,x-3,y] 
                                             + a_m2*vx[time+1,x-2,y]
                                             + a_m1*vx[time+1,x-1,y]
                                             + a_1*vx[time+1,x+1,y]
                                             + a_2*vx[time+1,x+2,y]
                                             + a_3*vx[time+1,x+3,y])/dx)


u_tyy_dev = Eq(tyy_dev.forward, tyy_dev + (rho*V_p**2)*dt * vy_dev.forward.dy + (rho*V_p**2-2*rho*V_s**2)*dt * vx_dev.forward.dx)

bc_txy = [Eq(txy[time+1,0,y], 0.0)]
bc_txy += [Eq(txy[time+1,300,y], 0.0)]
bc_txy += [Eq(txy[time+1,x,0], 0.0)]
bc_txy += [Eq(txy[time+1,x,300], 0.0)]

u_txy = Eq(txy.forward, txy + (rho*V_s**2)*dt * ((a_m3*vx[time+1,x,y-3]
                                                  + a_m2*vx[time+1,x,y-2]
                                                  + a_m1*vx[time+1,x,y-1]
                                                  + a_1*vx[time+1,x,y+1]
                                                  + a_2*vx[time+1,x,y+2]
                                                  + a_3*vx[time+1,x,y+3])/dy 
                                                 + (a_m3*vy[time+1,x-3,y] 
                                                    + a_m2*vy[time+1,x-2,y]
                                                    + a_m1*vy[time+1,x-1,y]
                                                    + a_1*vy[time+1,x+1,y]
                                                    + a_2*vy[time+1,x+2,y]
                                                    + a_3*vy[time+1,x+3,y])/dx))

u_txy_dev = Eq(txy_dev.forward, txy_dev + (rho*V_s**2)*dt * (vx_dev.forward.dy + vy_dev.forward.dx))

op = Operator([u_vx, u_vy, u_txx, u_tyy, u_txy] + src_xx + src_yy + bc_vx + bc_vy + bc_txx + bc_tyy + bc_txy)

op_dev = Operator([u_vx_dev, u_vy_dev, u_txx_dev, u_tyy_dev, u_txy_dev] + src_xx_dev + src_yy_dev)

vx.data[:] = 0. #Velocity components
vy.data[:] = 0.
txx.data[:] = 0. #Symmetric stress tensors
tyy.data[:] = 0.
txy.data[:] = 0.

op()

vx_dev.data[:] = 0. #Velocity components
vy_dev.data[:] = 0.
txx_dev.data[:] = 0. #Symmetric stress tensors
tyy_dev.data[:] = 0.
txy_dev.data[:] = 0.

op_dev()

plt.imshow(swapaxes(vx.data[1], 0, 1), interpolation='bicubic')#, vmin = -0.01, vmax = 0.01)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title('$v_x$ at t = %.2f ms; Optimized 4th order\nCourant number = %.2f' % (tn, 2500.*dt/dx))
plt.colorbar(label="Velocity ($ms^{-1}$)")
#plt.savefig("Figures/2D_Elastic/%.0f_vx_opt" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(vy.data[1], 0, 1), interpolation='bicubic')#, vmin = -0.01, vmax = 0.01)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title('$v_y$ at t = %.2f ms; Optimized 4th order\nCourant number = %.2f' % (tn, 2500.*dt/dx))
plt.colorbar(label="Velocity ($ms^{-1}$)")
#plt.savefig("Figures/2D_Elastic/%.0f_vy_opt" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(txx.data[1], 0, 1), interpolation='bicubic')#, vmin = -20., vmax = 20.)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title('${\\tau}_{xx}$ at t = %.2f ms; Optimized 4th order\nCourant number = %.2f' % (tn, 2500.*dt/dx))
plt.colorbar(label="Stress (Pa)")
#plt.savefig("Figures/2D_Elastic/%.0f_txx_opt" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(tyy.data[1], 0, 1), interpolation='bicubic')#, vmin = -20., vmax = 20.)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title('${\\tau}_{yy}$ at t = %.2f ms; Optimized 4th order\nCourant number = %.2f' % (tn, 2500.*dt/dx))
plt.colorbar(label="Stress (Pa)")
#plt.savefig("Figures/2D_Elastic/%.0f_tyy_opt" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(txy.data[1], 0, 1), interpolation='bicubic')#, vmin = -20., vmax = 20.)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title('${\\tau}_{xy}$ at t = %.2f ms; Optimized 4th order\nCourant number = %.2f' % (tn, 2500.*dt/dx))
plt.colorbar(label="Stress (Pa)")
#plt.savefig("Figures/2D_Elastic/%.0f_txy_opt" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(vx_dev.data[1], 0, 1), interpolation='bicubic')#, vmin = -0.01, vmax = 0.01)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title('$v_x$ at t = %.2f ms; Conventional 6th order\nCourant number = %.2f' % (tn, 2500.*dt/dx))
plt.colorbar(label="Velocity ($ms^{-1}$)")
#plt.savefig("Figures/2D_Elastic/%.0f_vx_dev" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(vy_dev.data[1], 0, 1), interpolation='bicubic')#, vmin = -0.01, vmax = 0.01)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title('$v_y$ at t = %.2f ms; Conventional 6th order\nCourant number = %.2f' % (tn, 2500.*dt/dx))
plt.colorbar(label="Velocity ($ms^{-1}$)")
#plt.savefig("Figures/2D_Elastic/%.0f_vy_dev" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(txx_dev.data[1], 0, 1), interpolation='bicubic')#, vmin = -20., vmax = 20.)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title('${\\tau}_{xx}$ at t = %.2f ms; Conventional 6th order\nCourant number = %.2f' % (tn, 2500.*dt/dx))
plt.colorbar(label="Stress (Pa)")
#plt.savefig("Figures/2D_Elastic/%.0f_txx_dev" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(tyy_dev.data[1], 0, 1), interpolation='bicubic')#, vmin = -20., vmax = 20.)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title('${\\tau}_{yy}$ at t = %.2f ms; Conventional 6th order\nCourant number = %.2f' % (tn, 2500.*dt/dx))
plt.colorbar(label="Stress (Pa)")
#plt.savefig("Figures/2D_Elastic/%.0f_tyy_dev" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(txy_dev.data[1], 0, 1), interpolation='bicubic')#, vmin = -20., vmax = 20.)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title('${\\tau}_{xy}$ at t = %.2f ms; Conventional 6th order\nCourant number = %.2f' % (tn, 2500.*dt/dx))
plt.colorbar(label="Stress (Pa)")
#plt.savefig("Figures/2D_Elastic/%.0f_txy_dev" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(vx.data[1]-vx_dev.data[1], 0, 1), interpolation='bicubic')#, vmin = -0.01, vmax = 0.01)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title('$v_x$ at t = %.2f ms; Difference\nCourant number = %.2f' % (tn, 2500.*dt/dx))
plt.colorbar(label="Velocity ($ms^{-1}$)")
#plt.savefig("Figures/2D_Elastic/%.0f_vx_dif" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(vy.data[1]-vy_dev.data[1], 0, 1), interpolation='bicubic')#, vmin = -0.01, vmax = 0.01)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title('$v_y$ at t = %.2f ms; Difference\nCourant number = %.2f' % (tn, 2500.*dt/dx))
plt.colorbar(label="Velocity ($ms^{-1}$)")
#plt.savefig("Figures/2D_Elastic/%.0f_vy_dif" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(txx.data[1]-txx_dev.data[1], 0, 1), interpolation='bicubic')#, vmin = -20., vmax = 20.)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title('${\\tau}_{xx}$ at t = %.2f ms; Difference\nCourant number = %.2f' % (tn, 2500.*dt/dx))
plt.colorbar(label="Stress (Pa)")
#plt.savefig("Figures/2D_Elastic/%.0f_txx_dif" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(tyy.data[1]-tyy_dev.data[1], 0, 1), interpolation='bicubic')#, vmin = -20., vmax = 20.)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title('${\\tau}_{yy}$ at t = %.2f ms; Difference\nCourant number = %.2f' % (tn, 2500.*dt/dx))
plt.colorbar(label="Stress (Pa)")
#plt.savefig("Figures/2D_Elastic/%.0f_tyy_dif" % (tn), dpi=200)
plt.show()

plt.imshow(swapaxes(txy.data[1]-txy_dev.data[1], 0, 1), interpolation='bicubic')#, vmin = -20., vmax = 20.)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title('${\\tau}_{xy}$ at t = %.2f ms; Difference\nCourant number = %.2f' % (tn, 2500.*dt/dx))
plt.colorbar(label="Stress (Pa)")
#plt.savefig("Figures/2D_Elastic/%.0f_txy_dif" % (tn), dpi=200)
plt.show()