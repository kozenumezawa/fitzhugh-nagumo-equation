# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import csv
import json

# to compare the calculation result with data calculated by the co-researcher
def compare(m, u):
    if m == 0:
        name = './data/Div50_000.0000.csv'
    if m == 50000:
        name = './data/Div50_005.0000.csv'
    elif m == 100000:
        name = './data/Div50_010.0000.csv'
    elif m == 150000:
        name = './data/Div50_015.0000.csv'
    elif m == 200000:
        name = './data/Div50_020.0000.csv'
    elif m == 250000:
        name = './data/Div50_025.0000.csv'
    elif m == 300000:
        name = './data/Div50_030.0000.csv'
    with open(name, 'r') as f:
        reader = csv.reader(f)
        sim_data = []
        for row in reader:
            sim_data.append(np.array(row, dtype=np.float))
        sim_data = np.array(sim_data)
        sim_data = sim_data[::-1]
        # print(sum(sum(abs(sim_data - u))))

# add noise to data for the experiment
# a_x = np.random.randint(0, 48)
# a_y = np.random.randint(0, 48)
# b_x = np.random.randint(0, 48)
# b_y = np.random.randint(0, 48)
# c_x = np.random.randint(0, 48)
# c_y = np.random.randint(0, 48)
# d_x = np.random.randint(0, 48)
# d_y = np.random.randint(0, 48)
a_x1 = 24
a_y1 = 15
a_x2 = 24
a_y2 = 30

b_x1 = 36
b_y1 = 15
b_x2 = 36
b_y2 = 30

c_x1 = 15
c_y1 = 15
c_x2 = 15
c_y2 = 30

def add_noise(all_time_series):
    # noise = np.random.rand(all_time_series.shape[0], all_time_series.shape[1]) / 10
    noise = np.zeros(shape=(all_time_series.shape[0], all_time_series.shape[1]))
    all_time_series_noise = all_time_series + noise
    return all_time_series_noise

if __name__ == "__main__":
    w = h = 50.
    # intervals in x-, y- directions, mm
    dx = dy = 1.
    # Thermal diffusivity of steel, mm2.s-1
    # D = 20.

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

    alpha = 0.1
    def do_timestep(u0, u):
        D = 20
        # Propagate with forward-difference in time, central-difference in space
        u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * (
            (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dy2
            + (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dx2 ) \
            + dt * u0[1:-1, 1:-1] * (1 - u0[1:-1, 1:-1]) * (u0[1:-1, 1:-1] - alpha)

        # neumann boundary condition
        u[0, 1:-1] = u0[0, 1:-1] + D * dt * (
            (u0[1, 1:-1] - 2*u0[0, 1:-1] + u0[0, 1:-1])/dy2
            + (u0[0, 2:] - 2*u0[0, 1:-1] + u0[0, :-2])/dx2 ) \
            + dt * u0[0, 1:-1] * (1 - u0[0, 1:-1]) * (u0[0, 1:-1] - alpha)

        u[49, 1:-1] = u0[49, 1:-1] + D * dt * (
            (u0[49, 1:-1] - 2*u0[49, 1:-1] + u0[48, 1:-1])/dy2
            + (u0[49, 2:] - 2*u0[49, 1:-1] + u0[49, :-2])/dx2 ) \
            + dt * u0[49, 1:-1] * (1 - u0[49, 1:-1]) * (u0[49, 1:-1] - alpha)

        u[1:-1, 0] = u0[1:-1, 0] + D * dt * (
            (u0[2:, 0] - 2*u0[1:-1, 0] + u0[:-2, 0])/dy2
            + (u0[1:-1, 1] - 2*u0[1:-1, 0] + u0[1:-1, 0])/dx2 ) \
            + dt * u0[1:-1, 0] * (1 - u0[1:-1, 0]) * (u0[1:-1, 0] - alpha)

        u[1:-1, 49] = u0[1:-1, 49] + D * dt * (
            (u0[2:, 49] - 2*u0[1:-1, 49] + u0[:-2, 49])/dy2
            + (u0[1:-1, 49] - 2*u0[1:-1, 49] + u0[1:-1, 48])/dx2 ) \
            + dt * u0[1:-1, 49] * (1 - u0[1:-1, 49]) * (u0[1:-1, 49] - alpha)

        u[0, 0] = u0[0, 0] + D * dt * (
            (u0[1, 0] - 2*u0[0, 0] + u0[0, 0])/dy2
            + (u0[0, 1] - 2*u0[0, 0] + u0[0, 0])/dx2 ) \
            + dt * u0[0, 0] * (1 - u0[0, 0]) * (u0[0, 0] - alpha)
        u[49, 0] = u0[49, 0] + D * dt * (
            (u0[49, 0] - 2*u0[49, 0] + u0[48, 0])/dy2
            + (u0[49, 1] - 2*u0[49, 0] + u0[49, 0])/dx2 ) \
            + dt * u0[49, 0] * (1 - u0[49, 0]) * (u0[49, 0] - alpha)

        u[0, 49] = u0[0, 49] + D * dt * (
            (u0[1, 49] - 2*u0[0, 49] + u0[0, 49])/dy2
            + (u0[0, 49] - 2*u0[0, 49] + u0[0, 48])/dx2 ) \
            + dt * u0[0, 49] * (1 - u0[0, 49]) * (u0[0, 49] - alpha)
        u[49, 49] = u0[49, 49] + D * dt * (
            (u0[49, 49] - 2*u0[49, 49] + u0[48, 49])/dy2
            + (u0[49, 49] - 2*u0[49, 49] + u0[49, 48])/dx2 ) \
            + dt * u0[49, 49] * (1 - u0[49, 49]) * (u0[49, 49] - alpha)

        x_left = a_x2 - 1
        x_right = a_x2 + 1
        y_left = a_y2 - 1
        y_right = a_y2 + 1
        D = 2.
        u[x_left:x_right, y_left:y_right] = u0[x_left:x_right, y_left:y_right] + D * dt * (
            (u0[x_left + 1:x_right + 1, y_left:y_right] - 2*u0[x_left:x_right, y_left:y_right] + u0[x_left - 1:x_right - 1, y_left:y_right])/dy2
            + (u0[x_left:x_right, y_left + 1:y_right + 1] - 2*u0[x_left:x_right, y_left:y_right] + u0[x_left:x_right, y_left - 1:y_right - 1])/dx2 ) \
                        + dt * u0[x_left:x_right, y_left:y_right] * (1 - u0[x_left:x_right, y_left:y_right]) * (u0[x_left:x_right, y_left:y_right] - alpha)
        u0 = u.copy()
        return u0, u

    # Number of timesteps
    nsteps = 300001
    # Output 4 figures at these timesteps
    mfig = [0, 100000, 200000, 300000]
    fignum = 0
    fig = plt.figure()

    all_time_series = []
    for m in range(nsteps):
        if m % 1000 == 0:
            u1d = u.flatten()   # serialize two dimensional data to one dimension
            all_time_series.append(u1d)

        if m in mfig:
            fignum += 1
            print(m, fignum)
            ax = fig.add_subplot(220 + fignum)
            im = ax.imshow(u.copy(), cmap=plt.get_cmap('hot'), vmin=Tcool,vmax=1)
            ax.set_axis_off()
            ax.set_title('{:.1f} ms'.format(m*dt*1000))
            # with open(str(m) + '.csv', 'w') as f:
            #     writer = csv.writer(f, lineterminator='\n')
            #     writer.writerows(u.tolist())

            compare(m, u)

        u0, u = do_timestep(u0, u)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    cbar_ax.set_xlabel('$T$ / K', labelpad=20)
    fig.colorbar(im, cax=cbar_ax)
    # plt.show()

    all_time_series = np.array(all_time_series)
    all_time_series = all_time_series.transpose()
    # print(all_time_series.shape) = (2500, 301)

    all_time_series_noise = add_noise(all_time_series)
    saveJSON = {
        'allTimeSeries': all_time_series_noise.tolist(), # reverse interpolate_list
        'width': w
    }
    f = open("./data/NagumoSimulation.json", "w")
    json.dump(saveJSON, f)
    f.close()