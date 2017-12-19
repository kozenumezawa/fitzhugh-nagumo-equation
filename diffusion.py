# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import csv

# to compare the calculation result with data calculated by the co-researcher
def compare(m, u):
    if m == 0:
        name = './data/dif_Div50_000.0000.csv'
    if m == 50000:
        name = './data/dif_Div50_005.0000.csv'
    elif m == 100000:
        name = './data/dif_Div50_010.0000.csv'
    elif m == 150000:
        name = './data/dif_Div50_015.0000.csv'
    elif m == 200000:
        name = './data/dif_Div50_020.0000.csv'
    elif m == 250000:
        name = './data/dif_Div50_025.0000.csv'
    with open(name, 'r') as f:
        reader = csv.reader(f)
        sim_data = []
        for row in reader:
            sim_data.append(np.array(row, dtype=np.float))
        sim_data = np.array(sim_data)
        sim_data = sim_data[::-1]
        print(sum(sum(abs(sim_data - u))))

if __name__ == "__main__":
    w = h = 50.
    # intervals in x-, y- directions, mm
    dx = dy = 1.
    # Thermal diffusivity of steel, mm2.s-1
    D = 20.

    Tcool, Thot = 0, 5

    nx, ny = int(w/dx), int(h/dy)
    dx2, dy2 = dx*dx, dy*dy
    dt = 0.0001 # 0.0000625= (dx2 * dy2 / (2 * D * (dx2 + dy2))) / 10 when D = 4

    u0 = Tcool * np.ones((ny, nx)) # shape of u = u[y][x]
    u = u0

    # Initial conditions - ring of inner radius r, width dr centred at (cx,cy) (mm)
    cx_l = 20
    cx_h = 30
    cy_l = 45
    cy_h = 50
    for x in range(cx_l, cx_h, 1):
        for y in range(cy_l, cy_h, 1):
            u0[y,x] = Thot

    def do_timestep(u0, u):
        # Propagate with forward-difference in time, central-difference in space
        u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * (
            (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dy2
            + (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dx2 ) \
            # + dt * u0[1:-1, 1:-1] * (1 - u0[1:-1, 1:-1]) * (u0[1:-1, 1:-1] - 0.1)

        # neumann boundary condition
        u[0, 1:-1] = u0[0, 1:-1] + D * dt * (
            (u0[1, 1:-1] - 2*u0[0, 1:-1] + u0[0, 1:-1])/dy2
            + (u0[0, 2:] - 2*u0[0, 1:-1] + u0[0, :-2])/dx2 )

        u[49, 1:-1] = u0[49, 1:-1] + D * dt * (
            (u0[49, 1:-1] - 2*u0[49, 1:-1] + u0[48, 1:-1])/dy2
            + (u0[49, 2:] - 2*u0[49, 1:-1] + u0[49, :-2])/dx2 )

        u[1:-1, 0] = u0[1:-1, 0] + D * dt * (
            (u0[2:, 0] - 2*u0[1:-1, 0] + u0[:-2, 0])/dy2
            + (u0[1:-1, 1] - 2*u0[1:-1, 0] + u0[1:-1, 0])/dx2 )

        u[1:-1, 49] = u0[1:-1, 49] + D * dt * (
            (u0[2:, 49] - 2*u0[1:-1, 49] + u0[:-2, 49])/dy2
            + (u0[1:-1, 49] - 2*u0[1:-1, 49] + u0[1:-1, 48])/dx2 )

        u[0, 0] = u0[0, 0] + D * dt * (
            (u0[1, 0] - 2*u0[0, 0] + u0[0, 0])/dy2
            + (u0[0, 1] - 2*u0[0, 0] + u0[0, 0])/dx2 )
        u[49, 0] = u0[49, 0] + D * dt * (
            (u0[49, 0] - 2*u0[49, 0] + u0[48, 0])/dy2
            + (u0[49, 1] - 2*u0[49, 0] + u0[49, 0])/dx2 )

        u[0, 49] = u0[0, 49] + D * dt * (
            (u0[1, 49] - 2*u0[0, 49] + u0[0, 49])/dy2
            + (u0[0, 49] - 2*u0[0, 49] + u0[0, 48])/dx2 )
        u[49, 49] = u0[49, 49] + D * dt * (
            (u0[49, 49] - 2*u0[49, 49] + u0[48, 49])/dy2
            + (u0[49, 49] - 2*u0[49, 49] + u0[49, 48])/dx2 )

        u0 = u.copy()
        return u0, u

    # Number of timesteps
    nsteps = 250001
    # Output 4 figures at these timesteps
    mfig = [0, 100000, 200000, 250000]
    fignum = 0
    fig = plt.figure()
    for m in range(nsteps):
        if m in mfig:
            fignum += 1
            print(m, fignum)
            ax = fig.add_subplot(220 + fignum)
            im = ax.imshow(u.copy(), cmap=plt.get_cmap('hot'), vmin=Tcool,vmax=1)
            ax.set_axis_off()
            ax.set_title('{:.1f} ms'.format(m*dt*1000))
            with open(str(m) + '.csv', 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerows(u.tolist())

            compare(m, u)

        u0, u = do_timestep(u0, u)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    cbar_ax.set_xlabel('$T$ / K', labelpad=20)
    fig.colorbar(im, cax=cbar_ax)
    plt.show()