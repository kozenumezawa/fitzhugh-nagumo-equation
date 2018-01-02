# calculate causal direction using cross correlation from each point to every point
def calc_all(all_time_series, max_lag, lag_step, window_size):
    import numpy as np
    import math
    import json

    corr_list = []
    lag_list = []

    for (row_idx, x) in enumerate(all_time_series):
        print(row_idx)
        row_corr = []
        row_lag = []

        for y in all_time_series:
            zero_lag_corr = np.corrcoef(x, y)[0][1]
            if math.isnan(zero_lag_corr):
                row_corr.append(0)
                row_lag.append(0)
                continue

            plus_lag_corr = []
            minus_lag_corr = []
            for lag in range(lag_step, max_lag, lag_step):
                # if corr(x, y[lag:]) > corr(x, y) : there is causality from x to y
                plus_lag_corr.append(np.corrcoef(x[:len(x) - lag], y[lag:])[0][1])
                minus_lag_corr.append(np.corrcoef(x[lag:], y[:len(y) - lag])[0][1])

            each_corr = [max(minus_lag_corr), zero_lag_corr, max(plus_lag_corr)]
            max_idx = each_corr.index(max(each_corr))

            if max_idx == 0:
                # when y -> x
                row_corr.append(0)
                row_lag.append(0)
            elif max_idx == 1:
                row_corr.append(max(each_corr))
                row_lag.append(0)
            else:
                # when x -> y
                row_corr.append(max(each_corr))
                row_lag.append(plus_lag_corr.index(max(plus_lag_corr)) + 1)
        corr_list.append(row_corr)
        lag_list.append(row_lag)

    f = open("./expdata/corr_list-" + str(window_size) + ".json", "w")
    json.dump(corr_list, f)
    f.close()

    f = open("./expdata/lag_list-" + str(window_size) + ".json", "w")
    json.dump(lag_list, f)
    f.close()
    return (corr_list, lag_list)


def is_sampling_point(idx, width, mean_step):
    import math
    x = idx % width
    y = math.floor(idx / width)
    if x % mean_step == 1 and y % mean_step == 0:
        return True
    return False
