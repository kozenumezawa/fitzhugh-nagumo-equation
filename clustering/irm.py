def infinite_relational_model(corr_matrix, lag_matrix, threshold, sampled_coords, window_size):
    import numpy as np
    import math
    import json
    import time
    import itertools as it
    from multiprocessing import cpu_count
    from microscopes.common.rng import rng
    from microscopes.common.relation.dataview import numpy_dataview
    from microscopes.models import bb as beta_bernoulli
    from microscopes.irm.definition import model_definition
    from microscopes.irm import model, runner, query
    from microscopes.kernels import parallel
    from microscopes.common.query import groups, zmatrix_heuristic_block_ordering, zmatrix_reorder

    cluster_matrix = []
    graph = []

    # calculate graph
    for row in corr_matrix:
        graph_row = []
        for corr in row:
            if corr < threshold:
                graph_row.append(False)
            else:
                graph_row.append(True)

        graph.append(graph_row)

    graph = np.array(graph, dtype=np.bool)

    graph_size = len(graph)

    # conduct Infinite Relational Model
    defn = model_definition([graph_size], [((0, 0), beta_bernoulli)])
    views = [numpy_dataview(graph)]
    prng = rng()

    nchains = cpu_count()
    latents = [model.initialize(defn, views, r=prng, cluster_hps=[{'alpha':1e-3}]) for _ in xrange(nchains)]
    kc = runner.default_assign_kernel_config(defn)
    runners = [runner.runner(defn, views, latent, kc) for latent in latents]
    r = parallel.runner(runners)

    start = time.time()
    # r.run(r=prng, niters=1000)
    # r.run(r=prng, niters=100)
    r.run(r=prng, niters=20)
    print ("inference took", time.time() - start, "seconds")

    infers = r.get_latents()
    clusters = groups(infers[0].assignments(0), sort=True)
    ordering = list(it.chain.from_iterable(clusters))

    z = graph.copy()
    z = z[ordering]
    z = z[:,ordering]

    corr_matrix = corr_matrix[ordering]
    corr_matrix = corr_matrix[:,ordering]

    lag_matrix = lag_matrix[ordering]
    lag_matrix = lag_matrix[:,ordering]

    cluster_sampled_coords = np.array(sampled_coords)
    cluster_sampled_coords = cluster_sampled_coords[ordering]

    response_msg = {
        'corrMatrix': corr_matrix.tolist(),
        'lagMatrix': lag_matrix.tolist(),
        'clusterMatrix': z.tolist(),
        'clusterSampledCoords': cluster_sampled_coords.tolist(),
        'nClusterList': [len(cluster) for cluster in clusters],
        'ordering': ordering,
    }
    f = open("./expdata/clustermatrix-" + str(window_size) + ".json", "w")
    json.dump(response_msg, f)
    f.close()

    return response_msg
