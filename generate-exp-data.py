# -*- coding: utf-8 -*-

# generate data for the experiment based on Fitzhugh-Nagumo Equation
# step1: generate multiple data when alpha is changed
# step2: select characteristic points randomly
# step3: add noise
# step4: calculate cross-correlation
# step5: clustering
# step6: save JSON

import numpy as np
import matplotlib.pyplot as plt
import csv
import json

from causalinference import allcrosscorr
from clustering import irm

def randomSampling():
    while True:
        idx = np.random.randint(0, 47)
        # the cause of (idx % 3 == 0) is calculation of substituting
        if idx % 3 == 0:
            return idx

def getCoordsList(n):
    coords_list = []
    while len(coords_list) < n:
        x = randomSampling()
        y = randomSampling()
        new_coords = [x, y]
        if not new_coords in coords_list:
            coords_list.append(new_coords)
    return coords_list

def add_noise(all_time_series, magnitude):
    if magnitude == 0:
        return all_time_series
    noise = np.random.rand(all_time_series.shape[0], all_time_series.shape[1]) / magnitude
    all_time_series_noise = all_time_series + noise
    return all_time_series_noise

def isSamplingPoint(idx):
    if int(idx % 3) == 1:
        return True
    return False

def removeUselessTimeSeries(all_time_series, width):
    sampled_all_time_series = []
    sampled_coords = []

    for (idx, time_series) in enumerate(all_time_series):
        x = int(idx % width)
        y = int(idx / width)
        if isSamplingPoint(x) and isSamplingPoint(y):
            sampled_coords.append({
                'idx': idx,
                'x': x,
                'y': y
            })
            sampled_all_time_series.append(time_series)
    return (np.array(sampled_all_time_series), sampled_coords)


def sort(corr_matrix, lag_matrix, cluster_matrix, cluster_sampled_coords, n_cluster_list, ordering):
    cluster_range_list = []
    end_idx = 0
    for n_cluster in n_cluster_list:
        start_idx = end_idx
        end_idx = start_idx + n_cluster
        cluster_range_list.append({
            'start': start_idx,
            'end': end_idx
        })

    adjacency_matrix = []
    for causal_cluster_range in cluster_range_list:
        height = causal_cluster_range['end'] - causal_cluster_range['start']
        row = []
        for effect_cluster_range in cluster_range_list:
            if effect_cluster_range == causal_cluster_range:
                row.append(False)
                continue
            # count the number of connections between two clusters
            causal_cnt = 0
            for causal_idx in range(causal_cluster_range['start'], causal_cluster_range['end']):
                for effect_idx in range(effect_cluster_range['start'], effect_cluster_range['end']):
                    if cluster_matrix[causal_idx][effect_idx] == True:
                        causal_cnt += 1
            width = effect_cluster_range['end'] - effect_cluster_range['start']
            area = width * height
            if causal_cnt > area * 0.9:
                row.append(True)
                continue
            row.append(False)
        adjacency_matrix.append(row)

    # count the number of source and target
    n_sources = [row.count(True) for row in adjacency_matrix]
    adjacency_matrix_t = map(list, zip(*adjacency_matrix))
    n_targets = [col.count(True) for col in adjacency_matrix_t]

    # calculate difference between nTargets and nSources
    n_diffs = [n_source - n_target for (n_source, n_target) in zip(n_sources, n_targets)]
    n_diffs = np.array(n_diffs)

    cluster_order = n_diffs.argsort()
    cluster_order = cluster_order[::-1]

    # get new index after sorting
    new_order = []
    for (new_cluster_idx, old_cluster_idx) in enumerate(cluster_order):
        start = cluster_range_list[old_cluster_idx]['start']
        end = cluster_range_list[old_cluster_idx]['end']
        for new_idx in range(start, end):
            new_order.append(new_idx)

    # update the order of matrix according to the sorting result
    corr_matrix = corr_matrix[new_order]
    corr_matrix = corr_matrix[:, new_order]

    lag_matrix = lag_matrix[new_order]
    lag_matrix = lag_matrix[:, new_order]

    cluster_matrix = cluster_matrix[new_order]
    cluster_matrix = cluster_matrix[:, new_order]
    cluster_sampled_coords = cluster_sampled_coords[new_order]
    ordering = ordering[new_order]
    n_cluster_list = n_cluster_list[cluster_order]

    sort_result = {
        'corrMatrix': corr_matrix.tolist(),
        'lagMatrix': lag_matrix.tolist(),
        'clusterMatrix': cluster_matrix.tolist(),
        'clusterSampledCoords': cluster_sampled_coords.tolist(),
        'nClusterList': n_cluster_list.tolist(),
        'ordering': ordering.tolist(),
    }
    return sort_result


if __name__ == "__main__":
    width = height = 50.
    # intervals in x-, y- directions, mm
    dx = dy = 1.

    Tcool, Thot = 0, 5

    nx, ny = int(width/dx), int(height/dy)
    dx2, dy2 = dx*dx, dy*dy
    dt = 0.0001 # 0.0000625= (dx2 * dy2 / (2 * D * (dx2 + dy2))) / 10 when D = 4

    # step1: generate multiple data when alpha is changed
    all_time_series_list = []
    alpha_list = [0.08, 0.09, 0.1]
    for alpha in alpha_list:
        f = open("./simdata/NagumoSimulation-alpha" + str(alpha) + ".json", "r")
        json_data = json.load(f)

        all_time_series = np.array(json_data['allTimeSeries'], dtype=np.float)
        all_time_series_list.append(all_time_series)

    # step2: select characteristic points randomly so that avoid same points
    coords_list = getCoordsList(4)
    low_points = [
        {
            'x': coords_list[0][0],
            'y': coords_list[0][1]
        },
        {
            'x': coords_list[1][0],
            'y': coords_list[1][1]
        }
    ]

    high_points = [
        {
            'x': coords_list[2][0],
            'y': coords_list[2][1]
        },
        {
            'x': coords_list[3][0],
            'y': coords_list[3][1]
        }
    ]

    # substitute
    all_time_series = all_time_series_list[1]
    for (low_point, high_point) in zip(low_points, high_points):
        low_idx = int(low_point['y'] * width + low_point['x'])
        all_time_series[low_idx: low_idx + 3, :] = all_time_series_list[0][low_idx: low_idx + 3, :]

        high_idx = int(high_point['y'] * width + high_point['x'])
        all_time_series[high_idx: high_idx + 3, :] = all_time_series_list[2][high_idx: high_idx + 3, :]

    # step3: add noise
    noise = 0
    all_time_series_noise = add_noise(all_time_series, 0)

    # step4: sampling and calculate cross-correlation
    max_lag = 30
    lag_step = 1
    window_size = 3
    sampled_all_time_series, sampled_coords = removeUselessTimeSeries(all_time_series_noise, width)

    corr_matrix, lag_matrix = allcrosscorr.calc_all(sampled_all_time_series,  max_lag, lag_step, window_size)
    # f = open("./expdata/corr_list-" + str(window_size) + ".json", "r")
    # corr_matrix = json.load(f)
    # f = open("./expdata/lag_list-" + str(window_size) + ".json", "r")
    # lag_matrix = json.load(f)

    # step5: clustering
    corr_matrix = np.array(corr_matrix, dtype=np.float)
    lag_matrix = np.array(lag_matrix, dtype=np.float)
    threshold = 0.7

    json_data = irm.infinite_relational_model(corr_matrix, lag_matrix, threshold, sampled_coords, window_size)

    # sort
    cluster_matrix = np.array(json_data['clusterMatrix'])
    cluster_sampled_coords = np.array(json_data['clusterSampledCoords'])
    n_cluster_list = np.array(json_data['nClusterList'])
    ordering = np.array(json_data['ordering'])

    sort_result = sort(corr_matrix, lag_matrix, cluster_matrix, cluster_sampled_coords, n_cluster_list, ordering)

    # step6: save json
    saveJSON = {
        'allTimeSeries': all_time_series_noise.tolist(), # reverse interpolate_list
        'width': width,
        'noise': noise,
        'lowPoints': {
            'alpha': alpha_list[0],
            'points': low_points
        },
        'highPoints': {
            'alpha': alpha_list[2],
            'points': high_points
        },

        'sampledAllTimeSeries': sampled_all_time_series.tolist(),
        'sampledCoords': sampled_coords,
        'corrMatrix': sort_result['corrMatrix'],
        'lagMatrix': sort_result['lagMatrix'],
        'clusterMatrix': sort_result['clusterMatrix'],
        'clusterSampledCoords': sort_result['clusterSampledCoords'],
        'nClusterList': sort_result['nClusterList'],
        'ordering': sort_result['ordering'],
    }

    f = open("./expdata/experiment_data.json", "w")
    json.dump(saveJSON, f)
    f.close()