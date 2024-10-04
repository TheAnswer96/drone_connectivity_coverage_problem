import time

import networkx as nx

from util import *
from util_algorithms import *


def alg_E_MEP(instance):
    # print("Algorithm Single Scenario: MEP\n")
    outputs = []
    # Result of the random instance
    G = instance["graph"]
    # nx.draw(G)
    # plt.show()
    intervals = instance["intervals"]
    # print("There are %d trajectories" % len(intervals))
    for interval in intervals:
        length = interval["length"]
        # print(" Length=%.2f" % length)
        for I in interval["interval"]:
            tower = I["tower"]
            inf = I["inf"]
            sup = I["sup"]
            # print("  T%d [%.2f, %.2f]" % (tower, inf, sup))
    # print("")
    # create_instance_set_cover(instance["intervals"])
    for i in range(len(intervals)):
        sol = {
            "eccentricity": -1,
            "used_intervals": []
        }
        source = "S" + str(i)
        length = round(intervals[i]["length"], 2)
        ecc = nx.eccentricity(G, source)
        depth = 1
        while depth <= ecc:
            bfs_tree = nx.bfs_tree(G, source, depth_limit=depth)
            bfs_nodes = bfs_tree.nodes()
            # print("distance: ", depth)
            # print("nodes in connectivity graph: ", bfs_nodes)
            cover, coverage = is_coverage(intervals[i], set(bfs_nodes))
            # print("*Exists feasible coverage: ", cover, "\n")
            if cover:
                sol = {
                    "eccentricity": depth,
                    "used_intervals": get_minimum_cover(coverage, length),
                }
                break
            depth = depth + 1
        output = {"trajectory": i, "solution": sol}
        outputs.append(output)

    return outputs


def alg_C_MTCP(instance):
    print("Algorithm Single Scenario: MEP\n")
    outputs = []
    # Result of the random instance
    G = instance["graph"]
    # nx.draw(G)
    # plt.show()
    intervals = instance["intervals"]
    print("There are %d trajectories" % len(intervals))
    for interval in intervals:
        length = interval["length"]
        print(" Length=%.2f" % length)
        for I in interval["interval"]:
            tower = I["tower"]
            inf = I["inf"]
            sup = I["sup"]
            print("  T%d [%.2f, %.2f]" % (tower, inf, sup))
    print("")
    # create_instance_set_cover(instance["intervals"])
    for i in range(len(intervals)):
        temp = []
        sol = {"eccentricity": -1, "used_intervals": []}
        source = "S" + str(i)
        length = round(intervals[i]["length"], 2)
        ecc = nx.eccentricity(G, source)
        depth = 1
        while depth <= ecc:
            bfs_tree = nx.bfs_tree(G, source, depth_limit=depth)
            bfs_nodes = bfs_tree.nodes()
            print("distance: ", depth)
            print("nodes in connectivity graph: ", bfs_nodes)
            cover, coverage = is_coverage(intervals[i], set(bfs_nodes))
            print("*Exists feasible coverage: ", cover, "\n")
            if cover:
                sol = {
                    "eccentricity": depth,
                    "used_intervals": get_minimum_cover(coverage, length),
                }
                temp.append(sol)
            depth = depth + 1

        if not temp == []:
            sol = min(temp, key=lambda x: (len(x["used_intervals"]), x["eccentricity"]))
        output = {"trajectory": i, "solution": sol}
        outputs.append(output)

    return outputs


def alg_OPT_MEPT(instance):
    start_time = time.time()

    G = instance["graph"]
    intervals = instance["intervals"]

    # Determine the min d to cover all trajectories
    min_d_vec = []
    bfs_nodes_vec = []

    for i in range(len(intervals)):

        source = "S" + str(i)
        # length = round(intervals[i]["length"], 2)
        ecc = nx.eccentricity(G, source)
        depth = 1
        while depth <= ecc:
            bfs_tree = nx.bfs_tree(G, source, depth_limit=depth)
            bfs_nodes = bfs_tree.nodes()
            # print("distance: ", depth)
            # print("nodes in connectivity graph: ", bfs_nodes)
            cover, coverage = is_coverage(intervals[i], set(bfs_nodes))
            # print("*Exists feasible coverage: ", cover, "\n")
            if cover:
                min_d_vec.append(depth)
                bfs_nodes_vec.append(bfs_nodes)
                break
            depth = depth + 1

    # print(min_d_vec)
    # eccentricity to return
    max_min_d = -1
    if len(min_d_vec) == len(intervals):
        max_min_d = max(min_d_vec)
    # print(f"The minimum d to cover all trajectories is {max_min_d}")
    bfs_nodes = set()
    for nodes in bfs_nodes_vec:
        for node in nodes:
            if isinstance(node, int):
                bfs_nodes.add(node)

    graph_nodes = set()
    for node in G.nodes():
        if isinstance(node, int):
            graph_nodes.add(node)

    diff_nodes = graph_nodes - bfs_nodes
    # print(f"Towers in the original graph = {graph_nodes}")
    # print(f"Only the following towers will be used = {bfs_nodes}")
    # print(f" -> The following towers will be neglected: {diff_nodes}")

    universe, collection, tower_ids = create_instance_set_cover(intervals, bfs_nodes)

    # print(f"Universe: {universe}")
    # print("Collection of subsets:")
    # for i in range(0, len(collection)):
    #     print(f"Subset {i}:", collection[i])

    result = solve_set_cover_OPT(universe, collection)
    # print("Selected subsets with index ", result)
    # for i in result:
    #     print(f" Subset {i}:", collection[i])
    #
    # print("Selected towers")
    # for i in result:
    #     print(f" T_{tower_ids[i]}")

    towers_out = set()
    for i in result:
        towers_out.add(f"T{tower_ids[i]}")

    # print("Selected unique towers")
    res = []
    for t in towers_out:
        # print(f" T_{t}")
        res.append(t)

    end_time = time.time()
    elapsed_time = end_time - start_time

    output = {
        "algorithm": "alg_OPT_MEPT",
        "elapsed_time": round(elapsed_time, 4),
        "eccentricity": max_min_d,
        "towers": towers_out,
        "total_towers": len(towers_out)
    }

    return output


def alg_E_SC_MEPT(instance):
    start_time = time.time()

    G = instance["graph"]
    intervals = instance["intervals"]

    # print(intervals)
    # exit()
    # Determine the min d to cover all trajectories
    min_d_vec = []
    bfs_nodes_vec = []
    for i in range(len(intervals)):
        source = "S" + str(i)
        # length = round(intervals[i]["length"], 2)
        ecc = nx.eccentricity(G, source)
        depth = 1
        while depth <= ecc:
            bfs_tree = nx.bfs_tree(G, source, depth_limit=depth)
            bfs_nodes = bfs_tree.nodes()
            # print("distance: ", depth)
            # print("nodes in connectivity graph: ", bfs_nodes)
            cover, coverage = is_coverage(intervals[i], set(bfs_nodes))
            # print("*Exists feasible coverage: ", cover, "\n")
            if cover:
                min_d_vec.append(depth)
                bfs_nodes_vec.append(bfs_nodes)
                break
            depth = depth + 1

    max_min_d = -1
    if len(min_d_vec) == len(intervals):
        max_min_d = max(min_d_vec)

    # print(f"The minimum d to cover all trajectories is {max_min_d}")
    bfs_nodes = set()
    for nodes in bfs_nodes_vec:
        for node in nodes:
            if isinstance(node, int):
                bfs_nodes.add(node)

    graph_nodes = set()
    for node in G.nodes():
        if isinstance(node, int):
            graph_nodes.add(node)

    diff_nodes = graph_nodes - bfs_nodes
    # print(f"Towers in the original graph = {graph_nodes}")
    # print(f"Only the following towers will be used = {bfs_nodes}")
    # print(f" -> The following towers will be neglected: {diff_nodes}")

    universe, collection, tower_ids = create_instance_set_cover(intervals, bfs_nodes)
    # print(f"Universe: {universe}")
    # print("Collection of subsets:")
    # for i in range(0, len(collection)):
    #     print(f"Subset {i}:", collection[i])

    result = solve_set_cover_APX(universe, collection)
    # print("Selected subsets with index ", result)

    towers_out = set()
    for res in result:
        idx = collection.index(res)
        towers_out.add(f"T{idx}")

    # print("Selected unique towers")
    res = []
    for t in towers_out:
        # print(f" T_{t}")
        res.append(t)

    end_time = time.time()
    elapsed_time = end_time - start_time

    output = {
        "algorithm": "alg_E_SC_MEPT",
        "elapsed_time": round(elapsed_time, 4),
        "eccentricity": max_min_d,
        # "used_intervals": result,
        "towers": towers_out,
        "total_towers": len(towers_out)
    }

    return output


def alg_E_T_MEPT(instance):
    start_time = time.time()

    result = []
    max_min_d = 0
    unique_towers = set()

    # Call single_minimum_eccentricity function
    outputs = alg_E_MEP(instance)

    for output in outputs:
        index = output["trajectory"]
        sol = output["solution"]
        max_min_d = max(max_min_d, sol["eccentricity"])
        for interval in sol["used_intervals"]:
            unique_towers.add(interval[2])
        # single min eccentricity solution, namely MEP
        # result.append({
        #     "trajectory": index,
        #     "solution": sol
        # })

    # Total number of unique towers used across all trajectories
    total_unique_towers = len(unique_towers)

    end_time = time.time()
    elapsed_time = end_time - start_time

    output = {
        "algorithm": "alg_E_T_MEPT",
        "elapsed_time": round(elapsed_time, 4),
        "eccentricity": max_min_d,
        "towers": unique_towers,
        "total_towers": total_unique_towers
    }

    return output
