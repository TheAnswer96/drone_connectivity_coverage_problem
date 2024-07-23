from util import is_zero, create_interval_graph, is_coverage, get_minimum_cover
import networkx as nx
import matplotlib.pyplot as plt

def single_minimum_eccentricity(instance):
    print("Algorithm Single Scenario: MEP\n")
    outputs = []
    # Result of the random instance
    G = instance["graph"]
    nx.draw(G)
    plt.show()
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
        sol = {"eccentricity": -1, "used_intervals": []}
        source = "S"+str(i)
        length = round(intervals[i]["length"], 2)
        ecc = nx.eccentricity(G, source)
        depth = 1
        while depth < ecc:
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
                break
            depth = depth + 1
        output = {"trajectory": i, "solution": sol}
        outputs.append(output)

    return outputs


def single_minimum_coverage(instance):
    # TODO

    output = {
        "result": -1
    }

    return output


def single_minimum_k_coverage(instance):
    # TODO

    output = {
        "result": -1
    }

    return output


def create_instance_set_cover(intervals):
    i = 0
    for interval in intervals:
        length = interval["length"]
        print(f"Trajectory T_{i} with interval [0, {length:.2f}] and {len(interval['interval'])} towers")
        endpoints = []
        for I in interval["interval"]:
            tower = I["tower"]
            inf = I["inf"]
            sup = I["sup"]
            endpoints.append((inf, 'start', tower))
            endpoints.append((sup, 'end', tower))
            print(f"  I_{tower} [{inf:.2f}, {sup:.2f}]")

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
        print(f"The whole interval can be split into {len(segments)} mini intervals")
        for start, end, active_towers in segments:
            # print(f"  I_{i}^{j} -> [{start:.2f}, {end:.2f}], towers {active_towers}")
            print(f"  I_{(i, j)} -> [{start:.2f}, {end:.2f}], towers {active_towers}")
            mini_interval = {
                "subscript": i,
                "superscript": j,
                "inf": start,
                "sup": end,
                "active_towers": active_towers
            }
            mini_intervals.append(mini_interval)
            j = j + 1

        print(f"The whole interval can be split as follows")

        for I in interval["interval"]:
            tower = I["tower"]

            for mini_interval in mini_intervals:
                active_towers = mini_interval["active_towers"]
                subscript = mini_interval["subscript"]
                superscript = mini_interval["superscript"]

                for at in active_towers:
                    if at == tower:
                        I["mini"].append((subscript, superscript))

        for I in interval["interval"]:
            tower = I["tower"]
            inf = I["inf"]
            sup = I["sup"]
            mini = I["mini"]
            print(f"  I_{tower} [{inf:.2f}, {sup:.2f}] -> {mini}")
            # for subscript, superscript in mini:
            #     print(f"      I_{subscript}^{superscript}")

        # Next iteration
        i = i + 1

        print()

    return intervals


def multiple_minimum_eccentricity_opt(instance):
    G = instance["graph"]
    intervals = create_instance_set_cover(instance["intervals"])

    output = {
        "result": -1
    }

    return output


def multiple_minimum_eccentricity_v1(instance):
    # TODO

    output = {
        "result": -1
    }

    return output


def multiple_minimum_eccentricity_v2(instance):
    # TODO

    output = {
        "result": -1
    }

    return output
