import matplotlib.pyplot as plt
import networkx as nx
import math
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import os
import time
import problem_gen as problem

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


def circle_segment_intersection(x0, y0, x1, y1, cx, cy, r):
    # Vector from (x0, y0) to (x1, y1)
    dx = x1 - x0
    dy = y1 - y0

    # Quadratic coefficients
    a = dx ** 2 + dy ** 2
    b = 2 * (dx * (x0 - cx) + dy * (y0 - cy))
    c = (x0 - cx) ** 2 + (y0 - cy) ** 2 - r ** 2

    # Calculate the discriminant
    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        # No intersection
        return []

    # Calculate the two points of intersection
    sqrt_discriminant = math.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)

    # Find the intersection points
    intersection_points = []

    for t in [t1, t2]:
        if 0 <= t <= 1:  # Check if the intersection is within the segment
            ix = x0 + t * dx
            iy = y0 + t * dy
            intersection_points.append((ix, iy))

    return intersection_points


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
    # print(towers)
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
        model.addConstr(gp.quicksum(x[j] for j in range(len(collection)) if element in collection[j]) >= 1,
                        name=f"Cover_{element}")

    # Optimize the model
    model.optimize()

    # Get the result
    selected_subsets = [j for j in range(len(collection)) if x[j].x > 0.5]

    return selected_subsets


def solve_set_cover_APX(universe, collection):
    # APX: greedy algorithm that selects at every step the collection with the maximum coverage until the universe is covered
    uncovered = universe.copy()
    selected_collections_index = []
    selected_collections = []

    # print("Initial Universe:", universe)
    # print("Initial Collection:", collection)

    while uncovered:
        best = None
        best_index = -1
        max_cover = 0
        for index, col in enumerate(collection):
            intersection = uncovered & col
            intersection_size = len(intersection)
            # print(f"Subset {index}: {col}, Covers {intersection_size} uncovered elements")
            if intersection_size > max_cover:
                max_cover = intersection_size
                best = col
                best_index = index
        if max_cover == 0:
            # print("No subset can cover any more uncovered elements.")
            break
        selected_collections_index.append(best_index)
        selected_collections.append(best)
        uncovered = uncovered - best
        # print(f"\nSelected Subset {best_index}: {best}")
        # print(f"Uncovered Elements Remaining: {uncovered}\n")

    # print("Selected Subsets Indices:", selected_collections_index)
    # print("Selected Subsets Collections:", selected_collections)
    return selected_collections


def experiments(iterations, hyperparams):
    if not os.path.exists("./exp"):
        print("exp directory creation.")
        os.makedirs("./exp")

    for i in range(1, iterations+1):
        print(f"Iteration {i}/{iterations}")
        # Input parameters
        config = {
            "area_side": hyperparams["area_side"],
            "towers": hyperparams["towers"],
            "radius_min": hyperparams["radius_min"],
            "radius_max": hyperparams["radius_max"],
            "trajectories": hyperparams["trajectories"],
            "min_dist_trajectory": hyperparams["min_dist_trajectory"],
            "scenario": hyperparams["scenario"],
            "lattice_neighbors": hyperparams["lattice_neighbors"],
            "star_edges": hyperparams["star_edges"],
            "seed": i,
            "debug": hyperparams["debug"]
        }

        # Random instance
        start_time = time.time()
        instance = problem.generate_problem_instance(config)
        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"generate_problem_instance execution time: {round(elapsed_time, 4)} s")

        # Algorithms
        if ALGORITHM == 0:
            # Minimum Eccentricity Problem - MEP
            output = alg_E_MEP(instance)
            print(output)
        # elif ALGORITHM == 1:
        #     # MEP-k
        #     output = single_minimum_k_coverage(instance)
        #     print(output)
            exit(1)
        elif ALGORITHM == 2:
            # Minimum Tower Coverage Problem - MTCP
            output = alg_C_MTCP(instance)
            print(output)
        elif ALGORITHM == 3:
            # Minimum Eccentricity Problem with multiple Trajectories - MEPT
            output = alg_OPT_MEPT(instance)
            print(output)
        elif ALGORITHM == 4:
            # Minimum Eccentricity Problem with multiple Trajectories - MEPT
            output = alg_E_SC_MEPT(instance)
            print(output)
        elif ALGORITHM == 5:
            # Minimum Eccentricity Problem with multiple Trajectories - MEPT
            output = alg_E_T_MEPT(instance)
            print(output)
        elif ALGORITHM == 6:
            output_v1 = alg_E_SC_MEPT(instance)
            print(output_v1)
            output_v2 = alg_E_T_MEPT(instance)
            print(output_v2)
            output_opt = alg_OPT_MEPT(instance)
            print(output_opt)
            print("########################")
    return