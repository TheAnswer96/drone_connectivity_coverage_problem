import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

'''
Write here the code related to problem instances generation
'''
# def generate_problem_instance(n_intervals, trajectory_length):
#     '''
#     Generates a problem instance with a specified number of intervals and trajectory length
#     n_intervals: number of intervals (towers)
#     trajectory_length: length of the drone path
#     '''
#     intervals = []
#     for _ in range(n_intervals):
#         start = random.randint(0, trajectory_length - 1)
#         end = random.randint(start, trajectory_length)
#         intervals.append((start, end))
#     return trajectory_length, intervals

def is_coverage(trajectory, intervals):
    '''
    Hint: the function check if the trajectory is covered by the intervals passed
    trajectory: length of the drone path
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

def can_cover_with_diameter(G, intervals, max_diameter):
    for nodes_subset in nx.connected_components(G):
        subgraph = G.subgraph(nodes_subset)
        if nx.diameter(subgraph) <= max_diameter:
            if is_coverage(max(max(interval) for interval in intervals), intervals):
                return True
    return False

def find_minimum_diameter_subgraph(towers, trajectory, edge_p, seed, debug=True):
    intervals, G = generate_problem_instance(towers, trajectory, edge_p, seed, debug)

    low, high = 2, nx.diameter(G)
    best_diameter = high

    while low <= high:
        mid = (low + high) // 2
        if can_cover_with_diameter(G, intervals, mid):
            best_diameter = mid
            high = mid - 1
        else:
            low = mid + 1

    return best_diameter

min_diameter = find_minimum_diameter_subgraph(towers, trajectory, edge_p, seed)
print("Minimum diameter of subgraph that can cover the trajectory:", min_diameter)

def algorithm_k_coverage(intervals, trajectory, k):
    """
    Algorithm 2: Check if every point of the trajectory is covered by at least k intervals

    Parameters:
    intervals : List of intervals [(s1, e1), (s2, e2), ...]
    trajectory (list): List of points in the trajectory
    k: Minimum number of intervals required to cover each point

    Returns:
    bool: True if every point of the trajectory is covered by at least k intervals, False otherwise
    """
    # Step 1: Get the endpoints of intervals and the observer
    P = set([s for s, _ in intervals] + [e for _, e in intervals] + [trajectory[0]])

    # Step 2: Sort P in increasing order
    P = sorted(list(P))

    # Step 3: Iterate over each point in P
    for p in P:
        # Step 4: Find the set of intervals intersecting with the line x = p
        I_p = [i for i, (s, e) in enumerate(intervals) if s <= p <= e]

        # Step 5: Create an empty list Q of size n
        Q = [None] * len(intervals)

        # Step 6: Iterate over each point in P
        for p in P:
            # Step 7: Check if p is the observer
            if p == trajectory[0]:
                # Step 8: Check if p is not covered by at least k intervals
                if len(I_p) < k:
                    return False

            # Step 10: Initialize r to 0
            r = 0

            # Step 11: Iterate over each interval in I_p
            for i in I_p:
                # Step 12: Check if the intersection point is the start of the interval
                if intervals[i][0] == p:
                    # Step 13: Add the interval to Q
                    Q[i] = intervals[i]
                # Step 14: Check if the intersection point is the end of the interval
                elif intervals[i][1] == p:
                    # Step 15: Remove the interval from Q
                    Q[i] = None
                    r += 1

            # Step 16: Check if the number of intervals in Q plus r is less than k
            if sum(1 for x in Q if x is not None) + r < k:
                return False

    # Step 17: Return True if every point of the trajectory is covered by at least k intervals
    return True




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
