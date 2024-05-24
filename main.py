import networkx as nx
import random
import matplotlib.pyplot as plt
from queue import Queue

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
    return  h

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

# if __name__ == '__main__':
#     # Example setup
#     n = 100  # Total number of units in the trajectory
#     min_units = 10  # Minimum units each tower must cover
#     num_towers = 5  # Number of towers
#     G = generate_graph(n)

    # Set the seed for reproducibility
    seed = 54

    # Generate drone trajectory and intervals
    drone_trajectory, intervals = generate_trajectory_intervals(G, seed)

    # Partition the trajectory using towers
    partitioned_intervals = partition_trajectory(n, min_units, num_towers)

    # Print the drone trajectory and intervals
    print(f"Drone trajectory: {drone_trajectory}")
    print(f"Intervals: {intervals}")

    # Partition the trajectory into intervals covered by towers
    intervals = partition_trajectory(n, min_units, num_towers)
    for i, (start, end) in enumerate(intervals):
        print(f"Tower {i+1} covers units from {start} to {end}")
