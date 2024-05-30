import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

'''
Write here the code related to problem instances generation
'''

def is_coverage(trajectory, intervals):
    '''
    Hint: the function check if the trajectory is covered by the intervals passed
    trajectory: lenght of the drone path
    intervals: list of tower coverages e.g., [start, end]
    '''
    set_tr = set()
    for i in range(trajectory + 1):
        set_tr.add(i)
    set_int = set()
    for interval in intervals:
        s, e = interval
        for i in range(s, e + 1):
            set_int.add(i)
    print("Double-check print [remove once the code is stable]")
    print("trajectory set: ", set_tr)
    print("intervals set:", set_int)
    return len(set_tr.difference(set_int)) == 0


def partition_trajectory(path, towers):
    '''
    How it works:
    1) Select randomly how many intervals will cover the trajectory
    2) Partition the trajectory regularly picking a random number of towers
    3) Randomly select a feasible offset for the starting (ending) point of each interval
    4) Add randomly the rest (if any)
    '''
    intervals = []

    n_towers_partitioning = np.random.randint(1, min(path, towers))
    size_inter = int(np.ceil(path / n_towers_partitioning))
    # create the set of intervals that cover the trajectory with random offset
    for i in range(n_towers_partitioning):
        start = i * size_inter
        end = min(start + size_inter, path)
        offset_start = 0
        if start != 0:
            offset_start = np.random.randint(0, start)
        offset_end = 0
        if path - end != 0:
            offset_end = np.random.randint(0, path - end)
        intervals.append([start - offset_start, end + offset_end])
    # create the rest
    if towers > n_towers_partitioning:
        n_rest = towers - n_towers_partitioning
        for i in range(n_rest):
            start = np.random.randint(0, path - 1)
            end = np.random.randint(start, path)
            intervals.append([start, end])
    print("Drone trajectory partitioned: ", towers, " intervals")
    print("The minimum coverage is: ", n_towers_partitioning, " towers")
    print("Intervals: ", intervals)
    coverage = is_coverage(path, intervals)
    print("Is there a coverage: ", coverage)
    return intervals

def generate_connectivity_graph(towers, edge_p, seed, graph_print=True):
    """
    Hint: function create a random graph with edge probability equal to "edge_p", then inserts the observer with random
    connections with the rest of the graph G.
    In addition, make a static merge with the intervals [node i connected to interval i]
    the function can be beautified...for the moment does the minimum.
    """
    nodes = np.arange(towers)
    G = nx.gnp_random_graph(towers, edge_p, seed)
    nx.set_node_attributes(G, [], "name")
    nx.set_node_attributes(G, "no", "observer")
    for index in range(len(G.nodes)):
        G.nodes[index]["interval"] = index #we can randomize this attribute
        print("vertex ", index, " is named ", G.nodes[index]["name"], " observer: ", G.nodes[index]["observer"])
    if graph_print:
        nx.draw(G)
        plt.show()
    n_observer_conn = np.random.randint(1, towers)
    connections = random.sample(sorted(nodes), n_observer_conn)
    G.add_node(towers)
    G.nodes[towers]["name"] = towers
    G.nodes[towers]["observer"] = "yes"
    for node in connections:
        G.add_edge(towers, node)
    if graph_print:
        nx.draw(G)
        plt.show()
    print(G)
    return G

def generate_problem_instance(towers, trajectory, edge_p, seed, debug=True):
    G = generate_connectivity_graph(towers, edge_p, seed, debug)
    intervals = partition_trajectory(trajectory, towers)
    return intervals, G