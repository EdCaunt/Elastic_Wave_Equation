from devito import Grid, Function, TimeFunction, NODE, Eq, Operator
from examples.seismic.source import RickerSource, TimeAxis
from numpy import sqrt
import matplotlib.pyplot as plt

extent = (150., 150.)
shape = (301, 301)

grid = Grid(extent=extent, shape=shape)

t0, tn = 0., 40. # Strange tiling effect at tn > 30
dt = (1. / sqrt(2.)) / 2500.
time_range = TimeAxis(start=t0, stop=tn, step=dt) #Set up time axis object for ricker source

src = RickerSource(name='src', grid=grid, f0=0.100, time_range=time_range) # Ricker wavelet source
src.coordinates.data[:] = [75., 75.] 

so = 2 # Spatial derivatives are second order accurate

x = grid.dimensions[0]
y = grid.dimensions[1]

#vx= TimeFunction(name='vx', grid=grid, staggered=x, space_order=so) 
#vy = TimeFunction(name='vy', grid=grid, staggered=y, space_order=so) 

#txx = TimeFunction(name='txx', grid=grid, staggered=NODE, space_order=so)
#tyy = TimeFunction(name='ty', grid=grid, staggered=NODE, space_order=so)
#txy = TimeFunction(name='txy', grid=grid, staggered=(x, y), space_order=so)

vx= TimeFunction(name='vx', grid=grid, space_order=so) 
vy = TimeFunction(name='vy', grid=grid, space_order=so) 

txx = TimeFunction(name='txx', grid=grid, space_order=so)
tyy = TimeFunction(name='ty', grid=grid, space_order=so)
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
#time = grid.time_dim
#help(grid)

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

#u_vx = Eq(vx.forward, vx + dt*rho*(txx.dx + txy.dy))

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

#u_vy = Eq(vy.forward, vy + dt*rho*(txy.dx + tyy.dy))

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


#u_txx = Eq(txx.forward, txx + (rho*V_p**2)*dt * vx.forward.dx + (rho*V_p**2-2*rho*V_s**2)*dt * vy.forward.dy)

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


#u_tyy = Eq(tyy.forward, tyy + (rho*V_p**2)*dt * vy.forward.dy + (rho*V_p**2-2*rho*V_s**2)*dt * vx.forward.dx)

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

#u_txy = Eq(txy.forward, txy + (rho*V_s**2)*dt * (vx.forward.dy + vy.forward.dx))

op = Operator([u_vx, u_vy, u_txx, u_tyy, u_txy] + src_xx + src_yy + bc_vx + bc_vy + bc_txx + bc_tyy + bc_txy)

vx.data[:] = 0. #Velocity components
vy.data[:] = 0.
txx.data[:] = 0. #Symmetric stress tensors
tyy.data[:] = 0.
txy.data[:] = 0.

op()

plt.imshow(vx.data[1], interpolation='bicubic')
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar()
plt.show()

plt.imshow(vy.data[1], interpolation='bicubic')
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar()
plt.show()

plt.imshow(txx.data[1], interpolation='bicubic')
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar()
plt.show()

plt.imshow(tyy.data[1], interpolation='bicubic')
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar()
plt.show()

plt.imshow(txy.data[1], interpolation='bicubic')
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar()
plt.show()