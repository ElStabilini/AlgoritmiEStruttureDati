import itertools
import pandas as pd
import numpy as np

from graphtools import *
from clustertools import *
from visualization import *

# compute internal_intercconnectivity
def internal_interconnectivity(graph, cluster):
    return np.sum(bisection_weights(graph, cluster))

#compute relatibe interconnectivity
def relative_interconnectivity(graph, cluster_i, cluster_j):
    edges = connecting_edges((cluster_i, cluster_j), graph)
    EC = np.sum(get_weights(graph, edges))
    ECci, ECcj = internal_interconnectivity(
        graph, cluster_i), internal_interconnectivity(graph, cluster_j)
    return EC / ((ECci + ECcj) / 2.0)

#compute internal closness
def internal_closeness(graph, cluster):
    cluster = graph.subgraph(cluster)
    edges = cluster.edges()
    weights = get_weights(cluster, edges)
    return np.sum(weights)

#compute relative closness
def relative_closeness(graph, cluster_i, cluster_j):
    edges = connecting_edges((cluster_i, cluster_j), graph)
    if not edges:
        return 0.0
    else:
        SEC = np.mean(get_weights(graph, edges))
    Ci, Cj = internal_closeness(
        graph, cluster_i), internal_closeness(graph, cluster_j)
    SECci, SECcj = np.mean(bisection_weights(graph, cluster_i)), np.mean(
        bisection_weights(graph, cluster_j))
    return SEC / ((Ci / (Ci + Cj) * SECci) + (Cj / (Ci + Cj) * SECcj))


def merge_score(g, ci, cj, a):
    return relative_interconnectivity(
        g, ci, cj) * np.power(relative_closeness(g, ci, cj), a)


def merge_best(graph, df, a, k, verbose=False):
    clusters = np.unique(df['cluster'])
    max_score = 0
    ci, cj = -1, -1
    if len(clusters) <= k:
        return False

    for combination in itertools.combinations(clusters, 2):
        i, j = combination
        if i != j:
            if verbose:
                print("Checking c%d c%d" % (i, j))
            gi = get_cluster(graph, [i])
            gj = get_cluster(graph, [j])
            edges = connecting_edges(
                (gi, gj), graph)
            if not edges:
                continue
            ms = merge_score(graph, gi, gj, a)
            if verbose:
                print("Merge score: %f" % (ms))
            if ms > max_score:
                if verbose:
                    print("Better than: %f" % (max_score))
                max_score = ms
                ci, cj = i, j

    if max_score > 0:
        if verbose:
            print("Merging c%d and c%d" % (ci, cj))
        df.loc[df['cluster'] == cj, 'cluster'] = ci
        for i, p in enumerate(graph.nodes()):
            if graph.node[p]['cluster'] == cj:
                graph.node[p]['cluster'] = ci
    return max_score > 0

''''
    BUILDING CLUSTER

    input:
    output: dataframe with cluster labels
'''

def cluster(df, k, knn=10, m=30, alpha=2.0, verbose=False, plot=False):
    graph = knn_graph(df, knn, verbose=True)
    graph = pre_part_graph(graph, m, df, verbose=True)
    iterm = tqdm(enumerate(range(m - k)), total=m-k)
    for i in iterm:
        merge_best(graph, df, alpha, k, verbose)
        if plot:
            plot2d_data(df)
    res = rebuild_labels(df)
    return res

'''
    REBUILDING LABELS OF THE DATA FRAME:

    input: dataframe
    1. builds a copy of the dataframe
    2. retrieves the unique values in the 'cluster' column of the DataFrame df. 
        It counts the occurrences of each cluster label using value_counts() function from pandas, then extracts the index of the resulting DataFrame, which represents the unique cluster labels
    3. Iterates over each unique cluster and assign the current value of c to all the elements of the original cluster
    output: copy of the dataframe with the new labels
'''

def rebuild_labels(df):
    ans = df.copy()
    clusters = list(pd.DataFrame(df['cluster'].value_counts()).index)
    c = 1
    for i in clusters:
        ans.loc[df['cluster'] == i, 'cluster'] = c
        c = c + 1
    return ans
