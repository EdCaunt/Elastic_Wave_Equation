import matplotlib.pyplot as plt
from devito import Grid, TimeFunction, Eq, solve, Operator

so_dev = 6
to_dev = 2
extent = (150., 150.)
shape = (151, 151)


dt = 0.2

grid = Grid(extent=extent, shape=shape)
t_end = 75 
ns = int(t_end/dt) # Number of timesteps = total time/timestep size.

# Set up function and stencil
u_dev = TimeFunction(name="u_dev", grid=grid, space_order=so_dev, time_order=to_dev)

u_dev.data[:] = 0.

time = grid.time_dim
x = grid.dimensions[0]
y = grid.dimensions[1]
h_x = grid.spacing[0]
h_y = grid.spacing[1]
# Optimized stencil coefficients
a_0 = -2.81299833
a_1 = 1.56808208
a_2 = -0.17723283
a_3 = 0.01564992

eq_dev = Eq(u_dev.dt2-(a_3*u_dev[time, x - 3, y]
                       + a_2*u_dev[time, x - 2, y]
                       + a_1*u_dev[time, x - 1, y]
                       + a_0*u_dev[time, x, y]
                       + a_1*u_dev[time, x + 1, y]
                       + a_2*u_dev[time, x + 2, y]
                       + a_3*u_dev[time, x + 3, y])/h_x**2
            -u_dev.dy2)


#eq_dev = Eq(u_dev.dt2-u_dev.dx2-u_dev.dy2)

stencil_dev = solve(eq_dev, u_dev.forward)

print(stencil_dev)
print(x, y, t)

# Create operator
op_dev = Operator([Eq(u_dev.forward, stencil_dev)])

# Apply operator
op_dev.apply(time_M=ns-1, dt=dt)

plt.imshow(u_dev.data[0])
plt.show()