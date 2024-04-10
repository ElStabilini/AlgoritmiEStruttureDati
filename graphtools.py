import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import metis #export METIS_DLL=/usr/local/lib/libmetis.so

'''
    GETTING EUCLIDEAN DISTANCE

    input: two points of the dataset (each point is a list)
    output: norm of the vector determined by the difference of two vectors (vectors built from the list of attributes of the data points)
'''

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

'''
    BUILDING kNN graph:

    input: dataframe, number of NN, boolean to indicate whther to print progress information during execution
    1. extract points for the dataframe (ciascun punto è una riga del dataframe e itero sulle tuple)
    2. build original graph with all the points (NO edges at this point, only nodes)
    3. iterates on each couple of points computing the euclidean distance: definisco una lambda function che prende un punto x come argomento e calcola la distanza
        tra p e x (sto sempre ciclando su p). distances is a list of a number of points element that contains the distances between
        each point p and another 
    4. closests: returns index of the k closests points
    5. add  edge between every pair of closests points (i,c) defining: WEIGHT and SIMILIARITY
        (similarity is an attribute)
    6. set the "position" attribute for each node
    7. set the attribute 'edge_weight_attr' of the graph to 'similarity'
'''
def knn_graph(df, k, verbose=False):
    points = [p[1:] for p in df.itertuples()]
    g = nx.Graph()
    for i in range(0, len(points)):
        g.add_node(i)
    if verbose:
        print("Building kNN graph (k = %d)..." % (k))
    iterpoints = tqdm(enumerate(points), total=len(
        points)) if verbose else enumerate(points)
    for i, p in iterpoints:
        distances = list(map(lambda x: euclidean_distance(p, x), points))
        closests = np.argsort(distances)[1:k+1]  # second trough kth closest
        # print(distances[0])
        for c in closests:
            g.add_edge(i, c, weight=1.0 / distances[c], similarity=int(
                1.0 / distances[c] * 1e4))
        g.node[i]['pos'] = p
    g.graph['edge_weight_attr'] = 'similarity'
    return g

'''
    PARTITIONING GRAPH:

    input: graph, number of partitions, dataframe

    output: input graph with an  additional attribute 'partition', which indicates the partition that
'''
def part_graph(graph, k, df=None):
    edgecuts, parts = metis.part_graph(
        graph, 2, objtype='cut', ufactor=250)
    # print(edgecuts)
    for i, p in enumerate(graph.nodes()):
        graph.node[p]['cluster'] = parts[i]
    if df is not None:
        df['cluster'] = nx.get_node_attributes(graph, 'cluster').values()
    return graph

'''
    GETTING CLUSTERS

    input: graph
    1. For each node n in the graph, it checks if the 'cluster' attribute of the node is in the clusters list. 
        This condition graph.node[n]['cluster'] in clusters filters out nodes that do not belong to any of the clusters specified in the clusters list
    2. Note that clusters is a list that cointains the clusters identifiers
    output: nodes (data-points) in the cluster
'''
def get_cluster(graph, clusters):
    nodes = [n for n in graph.node if graph.node[n]['cluster'] in clusters]
    return nodes

'''
    CONNECTING EDGES - costruzione dell'insieme di taglio

    input: graph and partitions (partitions is a list containing the nodes from two parts of a possible partition of the graph)
    1. inizializzo l'insieme di taglio ad una lista vuota
    2. per ogni nodo nella prima partizione e per ogni nodo nella seconda partizione
    3. check if the node a is in the grap and if in the grap the nod a is connected to b
        if it is I add the couple (a,b) to the cut set
    output: cut_set for a given partition
'''

def connecting_edges(partitions, graph):
    cut_set = []
    for a in partitions[0]:
        for b in partitions[1]:
            if a in graph:
                if b in graph[a]:
                    cut_set.append((a, b))
    return cut_set

'''
    DETERMING MIN_CUT BISECTOR

    input: graph
    1. makes a copy of the input graph
    2. partions the graph into two parts (see part_graph functions - returns the pgraph with ann extra attribute to the nodes which indicates to wich partition it belongs)
    3. partitions is a  list containing two sets of nodes each represents a partition
    output: min_cut bisector (by calling the connecting edges function that gives the cut set for the partition)
'''

def min_cut_bisector(graph):
    graph = graph.copy()
    graph = part_graph(graph, 2)
    partitions = get_cluster(graph, [0]), get_cluster(graph, [1])
    return connecting_edges(partitions, graph)

'''
    GETTING WEIGHTS

    input: graph, cut set of the min cut bisector
    output: list containing all the weights of the edges connecting the elements of the min cut bisector
'''

def get_weights(graph, edges):
    return [graph[edge[0]][edge[1]]['weight'] for edge in edges]

'''
    GETTING WEIGHTS FROM CLUSTER BISECTION

    input: graph and list of nodes in a cluster
    1. creates a new graph with only the nodes from the given cluster (and name this subgraph cluster)
    2. get the min cut bisector (edges è la lista dell'insieme di taglio minimo)
    3. get the weights of the min cut bisector
'''

def bisection_weights(graph, cluster):
    cluster = graph.subgraph(cluster)
    edges = min_cut_bisector(cluster)
    weights = get_weights(cluster, edges)
    return weights

