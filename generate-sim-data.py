# -*- coding: utf-8 -*-

# generate data for the experiment based on Fitzhugh-Nagumo Equation
# step1: generate multiple data when alpha is changed
# step2: select characteristic points randomly
# step3: add noise
# step4: save json

import numpy as np
import matplotlib.pyplot as plt
import csv
import json


def add_noise(all_time_series):
    # noise = np.random.rand(all_time_series.shape[0], all_time_series.shape[1]) / 10
    noise = np.zeros(shape=(all_time_series.shape[0], all_time_series.shape[1]))
    all_time_series_noise = all_time_series + noise
    return all_time_series_noise

if __name__ == "__main__":
    width = height = 50.
    # intervals in x-, y- directions, mm
    dx = dy = 1.

    Tcool, Thot = 0, 5

    nx, ny = int(width/dx), int(height/dy)
    dx2, dy2 = dx*dx, dy*dy
    dt = 0.0001 # 0.0000625= (dx2 * dy2 / (2 * D * (dx2 + dy2))) / 10 when D = 4

    # step1: generate multiple data when alpha is changed
    D = 20
    alpha_list = [0.08, 0.09, 0.1]
    for alpha in alpha_list:
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

        def do_timestep(u0, u, D, alpha):
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
            u0 = u.copy()
            return u0, u

        # Number of timesteps
        nsteps = 400001
        all_time_series = []
        for m in range(nsteps):
            if m % 100000 == 0:
                print(m)
            if m % 1000 == 0:
                u1d = u.flatten()   # serialize two dimensional data to one dimension
                all_time_series.append(u1d)
            u0, u = do_timestep(u0, u, D, alpha)

        all_time_series = np.array(all_time_series)
        all_time_series = all_time_series.transpose()

        saveJSON = {
            'allTimeSeries': all_time_series.tolist(), # reverse interpolate_list
            'width': width
        }
        f = open("./simdata/NagumoSimulation-alpha" + str(alpha) + ".json", "w")
        json.dump(saveJSON, f)
        f.close()
