import matplotlib.pyplot as plt
import networkx as nx
import math
import gurobipy as gp
from gurobipy import GRB

EPSILON = 1e-5  # Small epsilon to handle floating-point precision issues


def is_square(n):
    if n < 0:
        return False

    root = math.isqrt(n)
    return root * root == n


def is_zero(value):
    return abs(value) < EPSILON


def get_distance(p0, p1):
    return math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)


def do_intervals_overlap(interval1, interval2):
    start1, end1 = interval1
    start2, end2 = interval2
    return not (end1 < start2 or end2 < start1)


def create_interval_graph(instance):
    graphs = []
    intervals = instance["intervals"]
    # Add nodes for each interval
    for interval in intervals:
        length = interval["length"]
        G = nx.Graph()
        for i, I in enumerate(interval['interval']):
            G.add_node(i, interval=I)

        # Add edges between overlapping intervals
        for i in range(len(interval['interval'])):
            for j in range(i + 1, len(interval['interval'])):
                if do_intervals_overlap(
                        [round(interval['interval'][i]['inf'], 2), round(interval['interval'][i]['sup'], 2)],
                        [round(interval['interval'][j]['inf'], 2), round(interval['interval'][j]['sup'], 2)]):
                    G.add_edge(i, j)

        # dummy nodes: starting and ending
        G.add_node(-1, interval=[0, 0])
        G.add_node(len(interval['interval']) + 1, interval=[length, length])
        for i in range(len(interval['interval'])):
            if do_intervals_overlap([0, 0], [round(interval['interval'][i]['inf'], 2),
                                             round(interval['interval'][i]['sup'], 2)]):
                G.add_edge(-1, i)
            if do_intervals_overlap([math.floor(length), math.ceil(length)], [round(interval['interval'][i]['inf'], 0),
                                                                              round(interval['interval'][i]['sup'],
                                                                                    2)]):
                G.add_edge(len(interval['interval']) + 1, i)
        graphs.append(G)
    return graphs


def is_coverage(intervals, nodes):
    # nodes is a set of tower names
    length = round(intervals["length"], 2)
    towers = []
    for interval in intervals["interval"]:
        if int(interval["tower"]) in nodes:
            towers.append([round(interval["inf"], 2), round(interval["sup"], 2), "T" + str(interval["tower"])])
    towers.sort(key=lambda x: x[0])
    print(towers)
    last_covered = 0
    for start, end, _ in towers:
        if start > last_covered:
            return False, towers
        last_covered = max(last_covered, end)
        # print(last_covered, " ", length, " statified ", last_covered >= length)
        if last_covered >= length:
            return True, towers
    return False, towers


def get_minimum_cover(cover, length):
    sol = []
    last_covered = 0
    while last_covered < length:
        temp = [tower for tower in cover if tower[0] <= last_covered]
        temp.sort(key=lambda x: x[1])
        best_I = temp[-1]
        # print("List: ", temp, " BEST I: ", best_I)
        last_covered = max(last_covered, best_I[1])
        sol.append(best_I)
    return sol


def solve_set_cover(universe, collection):
    # Create a new model
    model = gp.Model("SetCover")
    model.setParam('OutputFlag', False)

    # Create variables: x[j] is 1 if subset j is in the cover, 0 otherwise
    x = model.addVars(len(collection), vtype=GRB.BINARY, name="x")

    # Set objective: minimize the number of subsets in the cover
    model.setObjective(gp.quicksum(x[j] for j in range(len(collection))), GRB.MINIMIZE)

    # Add constraints: each element in the universe must be covered by at least one subset
    for element in universe:
        model.addConstr(gp.quicksum(x[j] for j in range(len(collection)) if element in collection[j]) >= 1, name=f"Cover_{element}")

    # Optimize the model
    model.optimize()

    # Get the result
    selected_subsets = [j for j in range(len(collection)) if x[j].x > 0.5]

    return selected_subsets
