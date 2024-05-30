import networkx as nx
import random
import matplotlib.pyplot as plt
from queue import Queue
import numpy as np

######################################### HYPER-PARAMETERS #############################################################
TOWERS = 5
TRAJECTORY = 5
SEED = 1

def generate_graph(n):
    G = nx.gnp_random_graph(n)
    return G

def bfs_labeling(G, s):
    """Label nodes with their distance from source node s using BFS."""
    labeling = {node: float('inf') for node in G.nodes()}
    labeling[s] = 0
    q = Queue()
    q.put(s)
    while not q.empty():
        node = q.get()
        for neighbor in G.neighbors(node):
            if labeling[neighbor] == float('inf'):
                labeling[neighbor] = labeling[node] + 1
                q.put(neighbor)
    return labeling

def diameter(G):
    """Calculate the diameter of the graph G."""
    return nx.diameter(G)

def feasible_coverage(G, labeling, h):
    """Check if all nodes within distance h from some node in G are covered."""
    sub_G = []
    for node in G.nodes():
        if labeling[node] <= h:
            sub_G.append(node)
    #check if the intervals associated to the nodes in sub_G cover the trajectories

def partition_trajectory(n, min_units, num_towers):
        intervals = []
        current_position = 0
        units_covered = 0

        while units_covered < n and len(intervals) < num_towers:
            start = current_position
            max_units = min(n - current_position, n - units_covered)
            end = start + random.randint(min_units, max_units)
            intervals.append((start, end))
            units_covered += (end - start)
            current_position = end

        return intervals
    #     if labeling[node] > h:
    #         return False
    # return True

def min_dist_conn(G, s):
    """Find the minimum distance that ensures all nodes are covered."""
    labeling = bfs_labeling(G, s)
    i = 2
    j = 2 + diameter(G)
    h = abs(i + j) // 2
    while j >= i:
        if feasible_coverage(G,labeling, h):
            j = h
        else:
            i = h
        h = abs(i + j) // 2
    return h

#def generate_graph(n):
 #   return
#def generate_trajectory_intervals(G, seed):
 #   drone_trajectory = [0,100]
  #  intervals = [[0,24],[22,89]]
   # return drone_trajectory, intervals
#if __name__ == '__main__':
 #   print("")

# def generate_graph(n):
#     G = nx.gnp_random_graph(n)
#     return G

def generate_trajectory_intervals(G, seed):
    random.seed(seed)
    drone_trajectory = []
    intervals = [[], []]
    return drone_trajectory, intervals

def is_coverage(trajectory, intervals):
    '''
    Hint: the function check if the trajectory is covered by the intervals passed
    trajectory: lenght of the drone path
    intervals: list of tower coverages e.g., [start, end]
    '''
    set_tr = set()
    for i in range(trajectory+1):
        set_tr.add(i)
    set_int = set()
    for interval in intervals:
        s, e = interval
        for i in range(s, e+1):
            set_int.add(i)
    print("Double-check print [remove once the code is stable]")
    print("trajectory set: ", set_tr)
    print("intervals set:", set_int)
    return len(set_tr.difference(set_int)) == 0

def partition_trajectory(path=TRAJECTORY, towers=TOWERS):
    '''
    How it works:
    1) Select randomly how many intervals will cover the trajectory
    2) Partition the trajectory regularly picking a random number of towers
    3) Randomly select a feasible offset for the starting (ending) point of each interval
    4) Add randomly the rest (if any)
    '''
    intervals = []

    n_towers_partitioning = np.random.randint(1, min(path, towers))
    size_inter = int(np.ceil(path/n_towers_partitioning))
    #create the set of intervals that cover the trajectory with random offset
    for i in range(n_towers_partitioning):
        start = i * size_inter
        end = min(start + size_inter, path)
        offset_start = 0
        if start != 0:
            offset_start = np.random.randint(0, start)
        offset_end = 0
        if path-end != 0:
            offset_end = np.random.randint(0, path - end)
        intervals.append([start-offset_start, end+offset_end])
    #create the rest
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

if __name__ == '__main__':
    # Set the seed for reproducibility
    # seed = 54
    #
    # # Generate drone trajectory and intervals
    # drone_trajectory, intervals = generate_trajectory_intervals(seed)
    #
    # # Partition the trajectory using towers
    # partitioned_intervals = partition_trajectory(TRAJECTORY, TOWERS)
    #
    # # Print the drone trajectory and intervals
    # print(f"Drone trajectory: {drone_trajectory}")
    # print(f"Intervals: {intervals}")
    #
    # # Partition the trajectory into intervals covered by towers
    # intervals = partition_trajectory(n, min_units, num_towers)
    # for i, (start, end) in enumerate(intervals):
    #     print(f"Tower {i+1} covers units from {start} to {end}")
    partition_trajectory()
