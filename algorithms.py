from util import is_zero, create_interval_graph, is_coverage, get_minimum_cover, solve_set_cover, solve_set_cover_APX
import networkx as nx
import matplotlib.pyplot as plt


def single_minimum_eccentricity(instance):
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
        sol = {"eccentricity": -1, "used_intervals": []}
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


def single_minimum_coverage(instance):
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


def single_minimum_k_coverage(instance):
    # TODO

    output = {
        "result": -1
    }

    return output


def create_instance_set_cover(intervals, bfs_nodes):
    i = 0
    for interval in intervals:
        length = interval["length"]
        # print(f"Trajectory T_{i} with interval [0, {length:.2f}] and {len(interval['interval'])} towers")
        endpoints = []
        for I in interval["interval"]:
            tower = I["tower"]
            if tower not in bfs_nodes:
                continue

            inf = I["inf"]
            sup = I["sup"]
            endpoints.append((inf, 'start', tower))
            endpoints.append((sup, 'end', tower))
            # print(f"  I_{tower} [{inf:.2f}, {sup:.2f}]")

        endpoints.sort()
        active_intervals = set()
        previous_point = None
        segments = []

        for point, event_type, tower in endpoints:
            if previous_point is not None and active_intervals:
                if not is_zero(point - previous_point):
                    segments.append((previous_point, point, list(active_intervals)))

            if event_type == 'start':
                active_intervals.add(tower)
            elif event_type == 'end':
                active_intervals.remove(tower)

            previous_point = point

        j = 0
        mini_intervals = []
        # print(f"The whole interval can be split into {len(segments)} mini intervals")
        for start, end, active_towers in segments:
            # print(f"  I_{i}^{j} -> [{start:.2f}, {end:.2f}], towers {active_towers}")
            mini_interval = {
                "subscript": i,
                "superscript": j,
                "inf": start,
                "sup": end,
                "active_towers": active_towers
            }
            mini_intervals.append(mini_interval)
            j = j + 1

        # print(f"The whole interval can be split as follows")

        for I in interval["interval"]:
            tower = I["tower"]
            if tower not in bfs_nodes:
                continue

            for mini_interval in mini_intervals:
                active_towers = mini_interval["active_towers"]
                subscript = mini_interval["subscript"]
                superscript = mini_interval["superscript"]

                for at in active_towers:
                    if at == tower:
                        I["mini"].append((subscript, superscript))

        for I in interval["interval"]:
            tower = I["tower"]
            if tower not in bfs_nodes:
                continue

            inf = I["inf"]
            sup = I["sup"]
            mini = I["mini"]
            # print(f"  I_{tower} [{inf:.2f}, {sup:.2f}] -> {mini}")

        # Next iteration
        i = i + 1

        # print()

    universe = set()
    collection = []
    tower_ids = []
    for interval in intervals:
        for I in interval["interval"]:
            tower = I["tower"]
            if tower not in bfs_nodes:
                continue

            mini = I["mini"]
            tmp = set()
            for m in mini:
                universe.add(m)
                tmp.add(m)

            idx = tower_ids.index(tower) if tower in tower_ids else -1
            if idx != -1:  # If tower exists in the list
                collection[idx].update(tmp)  # Use update to merge sets
            else:
                collection.append(tmp)  # Append the set directly
                tower_ids.append(tower)

    return universe, collection, tower_ids


def multiple_minimum_eccentricity_opt(instance):
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

    result = solve_set_cover(universe, collection)
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

    output = {
        "eccentricity": max_min_d,
        "towers": towers_out,
        "total_towers": len(towers_out)
    }

    return output


def multiple_minimum_eccentricity_v1(instance):
    G = instance["graph"]
    intervals = instance["intervals"]

    print(intervals)
    exit()
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
    print(f"The minimum d to cover all trajectories is {max_min_d}")
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
    print(f"Towers in the original graph = {graph_nodes}")
    print(f"Only the following towers will be used = {bfs_nodes}")
    print(f" -> The following towers will be neglected: {diff_nodes}")

    universe, collection = create_instance_set_cover(intervals, bfs_nodes)
    print(f"Universe: {universe}")
    print("Collection of subsets:")
    for i in range(0, len(collection)):
        print(f"Subset {i}:", collection[i])

    result = solve_set_cover_APX(universe, collection)
    print("Selected subsets with index ", result)

    # retrieve the intervals/towers from the output subsets
    # I_4 [0.00, 446.18]
    # I_5 [372.18, 590.47]
    # I_1 [0.00, 458.97]
    # I_2 [6.88, 511.60]


    output = {
        "eccentricity": max_min_d,
        "used_intervals": result,
    }

    return output


def multiple_minimum_eccentricity_v2(instance):
    result = []
    max_min_d = 0
    unique_towers = set()

    # Call single_minimum_eccentricity function
    outputs = single_minimum_eccentricity(instance)

    for output in outputs:
        index = output["trajectory"]
        sol = output["solution"]
        max_min_d = max(max_min_d, sol["eccentricity"])
        for interval in sol["used_intervals"]:
            unique_towers.add(interval[2])
        #single min eccentricity solution, namely MEP
        # result.append({
        #     "trajectory": index,
        #     "solution": sol
        # })

    # Total number of unique towers used across all trajectories
    total_unique_towers = len(unique_towers)

    output = {
        "eccentricity": max_min_d,
        "towers": unique_towers,
        "total_towers": total_unique_towers
    }

    return output

