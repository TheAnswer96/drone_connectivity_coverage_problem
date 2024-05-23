import networkx as nx
import random
import matplotlib.pyplot as plt
from queue import Queue

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
            # Calculate the maximum possible units that can be covered in this interval
            remaining_units = n - units_covered
            max_units = min(n - current_position, remaining_units)

            if max_units < min_units:
                # If the remaining units are less than min_units, we need to adjust min_units
                interval_length = max_units
            else:
                interval_length = random.randint(min_units, max_units)

            end = start + interval_length
            intervals.append((start, end))
            units_covered += interval_length
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

def generate_graph(n):
    return
def generate_trajectory_intervals(G, seed):
    drone_trajectory = [0,100]
    intervals = [[0,24],[22,89]]
    return drone_trajectory, intervals
if __name__ == '__main__':
    print("")
