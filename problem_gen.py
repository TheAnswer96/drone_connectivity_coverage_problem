import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
# import sympy as sp
import math
from sympy import Polygon, Point

from util import get_distance, EPSILON, circle_segment_intersection, is_zero


def generate_problem_instance(config):
    area_side = config["area_side"]
    towers = config["towers"]
    radius_min = config["radius_min"]
    radius_max = config["radius_max"]
    trajectories = config["trajectories"]
    min_dist_trajectory = config["min_dist_trajectory"]
    scenario = config["scenario"]
    lattice_neighbors = config["lattice_neighbors"]
    star_edges = config["star_edges"]
    seed = config["seed"]
    debug = config["debug"]

    random.seed(seed)

    # Area
    points = [
        (0, 0),
        (area_side, 0),
        (area_side, area_side),
        (0, area_side),
        (0, 0)
    ]
    area_x_coords = [point[0] for point in points]
    area_y_coords = [point[1] for point in points]

    # Towers
    tower_points = []
    tower_radii = []
    G = nx.Graph()

    if scenario == -1:
        tower_points, tower_radii, G = create_test(config)
    elif scenario == 1:
        tower_points, tower_radii, G = create_RGG_fixed_radius(radius_min, towers, area_side)
    elif scenario == 2:
        tower_points, tower_radii, G = create_RGG_variable_radius(radius_min, radius_max, towers, area_side)
    elif scenario == 3:
        tower_points, tower_radii, G = create_regular_manhattan(towers, area_side)
    elif scenario == 4:
        tower_points, tower_radii, G = create_regular_diagonal(towers, area_side)
    elif scenario == 5:
        tower_points, tower_radii, G = create_bus(towers, area_side)
    elif scenario == 6:
        tower_points, tower_radii, G = create_ring_lattice(towers, lattice_neighbors, area_side)
    elif scenario == 7:
        tower_points, tower_radii, G = create_star(towers, star_edges, area_side)
    else:
        print("Not implemented yet!")
        exit(1)

    # if scenario == -1:
    #     trajectories_paths = []
    #     trajectories_paths.append([(120, 305), (975, 100)])
    #     trajectories_paths.append([(0, 449.999), (1000, 450)])

    dummy = set()

    # # ----------------------------------
    # # Plot
    # plt.figure(figsize=(10, 10))
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.grid(True)
    #
    # plt.xlim([0, area_side])
    # plt.ylim([0, area_side])
    # plt.tight_layout()
    #
    # # Area
    # plt.plot(area_x_coords, area_y_coords)
    # plt.fill(area_x_coords, area_y_coords, alpha=0.025)
    #
    # # Towers + connectivity + coverage
    # pos = nx.get_node_attributes(G, 'pos')
    # x = [pos[node][0] for node in G.nodes() if node not in dummy]
    # y = [pos[node][1] for node in G.nodes() if node not in dummy]
    # plt.scatter(x, y, color='orange')
    # for edge in G.edges():
    #     if edge[0] not in dummy and edge[1] not in dummy:
    #         x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
    #         y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
    #         plt.plot(x_coords, y_coords, color='black')
    #
    # for node, (x_coord, y_coord) in pos.items():
    #     plt.text(x_coord + 10, y_coord + 10, 'T' + str(node), fontsize=12, ha='right')
    #
    # for i in range(0, towers):
    #     tower_x = tower_points[i][0]
    #     tower_y = tower_points[i][1]
    #     radius = tower_radii[i]
    #
    #     circle = plt.Circle((tower_x, tower_y), radius, color='orange', alpha=0.1)
    #     plt.gca().add_patch(circle)
    #
    # plt.show()
    # exit(-44)
    # # ----------------------------------

    # Trajectories
    trajectories_paths = []
    intervals = []
    intersection_points = []
    i = 0
    while i < trajectories:
        # Generate first the trajectory
        x_0 = random.uniform(0, area_side)
        y_0 = random.uniform(0, area_side)
        x_1 = random.uniform(0, area_side)
        y_1 = random.uniform(0, area_side)
        dist = math.sqrt((x_1 - x_0) ** 2 + (y_1 - y_0) ** 2)

        # If short, discard
        if dist < min_dist_trajectory:
            continue

        # Calculate intersections
        # p0 = sp.Point(x_0, y_0)
        # p1 = sp.Point(x_1, y_1)
        p0 = (x_0, y_0)
        p1 = (x_1, y_1)

        intersections = []
        for j in range(0, towers):
            tower_x, tower_y = tower_points[j]
            radius = tower_radii[j]
            # segment = sp.Segment(p0, p1)
            # circle = sp.Circle(sp.Point(tower_x, tower_y), radius)
            circle_center = (tower_x, tower_y)
            # ints = (segment.intersection(circle))
            ints = circle_segment_intersection(x_0, y_0, x_1, y_1, tower_x, tower_y, radius)

            # for idx in range(0, len(ints)):
            #     v1_0 = ints[idx][0].evalf()
            #     v1_1 = ints[idx][1].evalf()
            #     v2_0 = ints2[idx][0]
            #     v2_1 = ints2[idx][1]
            #
            #     diff_0 = v1_0 - v2_0
            #     diff_1 = v1_1 - v2_1
            #
            #     if not (is_zero(diff_0) and is_zero(diff_1)):
            #         print("Error")

            if ints:
                if len(ints) == 2:
                    intersections.append((ints, j))
                else:
                    # Check distances
                    if get_distance(p0, circle_center) < radius:
                        ints.append(p0)
                    elif get_distance(p1, circle_center) < radius:
                        ints.append(p1)
                    else:
                        # Tangent case, empty interval, so skip
                        pass  # No action needed for this case

                    # if p0.distance(circle.center) < radius:
                    #     ints.append(p0)
                    # elif p1.distance(circle.center) < radius:
                    #     ints.append(p1)
                    # else:
                    #     # Tangent case, empty interval, so skip
                    #     break

                    intersections.append((ints, j))

        # Evaluate if it is fully covered
        dist = get_distance((x_0, y_0), (x_1, y_1))
        # print("P%d = [0, %.2f]" % (i, dist))
        path_intervals = []
        for j in range(0, len(intersections)):
            ints = intersections[j]
            tower_id = ints[1]

            # i0x = ints[0][0].x
            # i0y = ints[0][0].y
            # i1x = ints[0][1].x
            # i1y = ints[0][1].y
            i0x = ints[0][0][0]
            i0y = ints[0][0][1]
            i1x = ints[0][1][0]
            i1y = ints[0][1][1]
            dist_i0 = get_distance((x_0, y_0), (i0x, i0y))
            dist_i1 = get_distance((x_0, y_0), (i1x, i1y))

            interval = {
                "tower": tower_id,
                "inf": min(dist_i0, dist_i1),
                "sup": max(dist_i0, dist_i1),
                "mini": []
            }
            path_intervals.append(interval)

            # print("  T%d - [%.2f, %.2f]" % (tower_id, min(dist_i0, dist_i1), max(dist_i0, dist_i1)))

        path_intervals = sorted(path_intervals, key=lambda x: (x["inf"], x["sup"]))
        covered = is_covered(dist, path_intervals)

        # If not covered, discard
        if not covered:
            continue

        # Add two dummy nodes
        s_node = "S%d" % i
        d_node = "D%d" % i
        G.add_node(s_node)
        G.add_node(d_node)
        dummy.add(s_node)
        dummy.add(d_node)

        # Add dummy edges
        for j in range(0, towers):
            tower_x, tower_y = tower_points[j]
            radius = tower_radii[j]

            dist_i0j = get_distance((x_0, y_0), (tower_x, tower_y))
            dist_i1j = get_distance((x_1, y_1), (tower_x, tower_y))

            if dist_i0j <= radius:
                G.add_edge(s_node, j)

            if dist_i1j <= radius:
                G.add_edge(d_node, j)

        intersection_points.append(intersections)
        out_int = {
            "length": dist,
            "interval": path_intervals
        }
        intervals.append(out_int)

        trajectories_paths.append([(x_0, y_0), (x_1, y_1)])
        i = i + 1

    if debug:
        # Plot
        plt.figure(figsize=(10, 10))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)

        plt.xlim([0, area_side])
        plt.ylim([0, area_side])
        plt.tight_layout()

        # Area
        plt.plot(area_x_coords, area_y_coords)
        plt.fill(area_x_coords, area_y_coords, alpha=0.025)

        # Towers + connectivity + coverage
        pos = nx.get_node_attributes(G, 'pos')
        x = [pos[node][0] for node in G.nodes() if node not in dummy]
        y = [pos[node][1] for node in G.nodes() if node not in dummy]
        plt.scatter(x, y, color='orange')
        for edge in G.edges():
            if edge[0] not in dummy and edge[1] not in dummy:
                x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
                y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
                plt.plot(x_coords, y_coords, color='black')

        for node, (x_coord, y_coord) in pos.items():
            plt.text(x_coord + 10, y_coord + 10, 'T' + str(node), fontsize=12, ha='right')

        for i in range(0, towers):
            tower_x = tower_points[i][0]
            tower_y = tower_points[i][1]
            radius = tower_radii[i]

            circle = plt.Circle((tower_x, tower_y), radius, color='orange', alpha=0.1)
            plt.gca().add_patch(circle)

        # Trajectories
        for i in range(len(trajectories_paths)):
            x_0, y_0 = trajectories_paths[i][0]
            x_1, y_1 = trajectories_paths[i][1]

            plt.scatter(x_0, y_0, marker='*', color='red')
            plt.scatter(x_1, y_1, marker='o', color='green')

            x_values = [x_0, x_1]
            y_values = [y_0, y_1]

            plt.text(x_0 + 10, y_0 + 10, 'P' + str(i), fontsize=12)
            plt.plot(x_values, y_values)

        # Intersections
        for i in range(0, len(intersection_points)):
            intersections = intersection_points[i]
            x_0, y_0 = trajectories_paths[i][0]
            x_1, y_1 = trajectories_paths[i][1]
            dist = get_distance((x_0, y_0), (x_1, y_1))
            # print("P%d = [0, %.2f]" % (i, dist))
            # print(" S=(%.2f, %.2f)" % (x_0, y_0))
            # print(" D=(%.2f, %.2f)" % (x_1, y_1))

            for j in range(0, len(intersections)):
                ints = intersections[j]
                tower_id = ints[1]
                x_values = []
                y_values = []
                # print("  I -> T%d" % tower_id)

                # i0x = ints[0][0].x
                # i0y = ints[0][0].y
                # i1x = ints[0][1].x
                # i1y = ints[0][1].y
                i0x = ints[0][0][0]
                i0y = ints[0][0][1]
                i1x = ints[0][1][0]
                i1y = ints[0][1][1]
                tx = tower_points[tower_id][0]
                ty = tower_points[tower_id][1]

                x_values.append(i0x)
                y_values.append(i0y)
                # print("   (%.2f, %.2f)" % (i0x, i0y))
                dist_i0 = get_distance((x_0, y_0), (i0x, i0y))
                # print(dist_i0)
                plt.plot(i0x, i0y, marker='x')

                x_values.append(tx)
                y_values.append(ty)

                x_values.append(i1x)
                y_values.append(i1y)
                # print("   (%.2f, %.2f)" % (i1x, i1y))
                dist_i1 = get_distance((x_0, y_0), (i1x, i1y))
                # print(dist_i1)
                plt.plot(i1x, i1y, marker='x')

                # print("  T%d - [%.2f, %.2f]" % (tower_id, min(dist_i0, dist_i1), max(dist_i0, dist_i1)))

                plt.plot(x_values, y_values, color='blue', linestyle='dashed', linewidth=0.7)

        plt.show()

    output = {
        "graph": G,
        "intervals": intervals
    }

    return output


def create_test(config):
    area_side = config["area_side"]
    towers = config["towers"]
    radius_min = config["radius_min"]
    radius_max = config["radius_max"]
    trajectories = config["trajectories"]
    min_dist_trajectory = config["min_dist_trajectory"]
    scenario = config["scenario"]
    lattice_neighbors = config["lattice_neighbors"]
    seed = config["seed"]
    debug = config["debug"]

    tower_points = []
    tower_points.append([200, 200])
    tower_points.append([550, 140])
    tower_points.append([900, 200])
    tower_points.append([300, 300])
    tower_points.append([750, 250])

    tower_radii = []
    tower_radii.append(150)
    tower_radii.append(150)
    tower_radii.append(150)
    tower_radii.append(150)
    tower_radii.append(150)

    G = nx.Graph()
    for i in range(0, towers):
        G.add_node(i, pos=tower_points[i])
    for i in range(0, towers):
        for j in range(i + 1, towers):
            distance = math.sqrt(
                (tower_points[i][0] - tower_points[j][0]) ** 2 + (tower_points[i][1] - tower_points[j][1]) ** 2)
            if distance <= 150:
                G.add_edge(i, j)

    return tower_points, tower_radii, G


def is_covered(dist, intervals):
    if len(intervals) == 0:
        return False

    # Step 1: Extract and sort intervals based on "inf" and "sup"
    sorted_intervals = sorted(intervals, key=lambda x: (x["inf"], x["sup"]))

    # Step 2: Merge overlapping and contiguous intervals
    min_inf = sorted_intervals[0]["inf"]
    max_sup = sorted_intervals[0]["sup"]

    # Do this when there is only one interval
    if len(intervals) == 1:
        if dist - max_sup > EPSILON:
            return False
        if min_inf > EPSILON:
            return False

    for i in range(1, len(sorted_intervals)):

        # There is a gap immediately, so exit
        if min_inf > EPSILON:
            return False

        inf = sorted_intervals[i]["inf"]
        sup = sorted_intervals[i]["sup"]

        if inf - max_sup > EPSILON:
            # There is a gap, so exit
            return False
        else:
            max_sup = max(max_sup, sup)

    if dist - max_sup > EPSILON:
        # There is a gap at the end, so exit
        return False

    return True


def create_RGG_fixed_radius(radius, towers, area_side):
    max_attempts = 1000
    att = 0
    while att < max_attempts:
        # print("Attempt n %d" % att)
        # print(att)
        tower_points = []
        for i in range(0, towers):
            x = random.uniform(0, area_side)
            y = random.uniform(0, area_side)
            tower_points.append([x, y])

        # for each tower, generate randomly a radius within "radius_min" and "radius_max"
        tower_radii = []
        for i in range(0, towers):
            tower_radii.append(radius)

        G = nx.Graph()
        for i in range(0, towers):
            G.add_node(i, pos=tower_points[i])
        for i in range(0, towers):
            for j in range(i + 1, towers):
                distance = math.sqrt(
                    (tower_points[i][0] - tower_points[j][0]) ** 2 + (tower_points[i][1] - tower_points[j][1]) ** 2)
                if distance <= radius:
                    G.add_edge(i, j)

        is_connected = nx.is_connected(G)

        if is_connected:
            print("Attempt: %d - The graph G is connected." % att)
            return tower_points, tower_radii, G
        else:
            att = att+1

    print("The graph G is not connected.")
    exit(-1)


def create_RGG_variable_radius(radius_min, radius_max, towers, area_side):
    max_attempts = 1000
    att = 0
    while att < max_attempts:
        tower_points = []
        for i in range(0, towers):
            x = random.uniform(0, area_side)
            y = random.uniform(0, area_side)
            tower_points.append([x, y])

        # for each tower, generate randomly a radius within "radius_min" and "radius_max"
        tower_radii = []
        for i in range(0, towers):
            radius = random.uniform(radius_min, radius_max)
            tower_radii.append(radius)

        G = nx.Graph()
        for i in range(0, towers):
            G.add_node(i, pos=tower_points[i])
        for i in range(0, towers):
            for j in range(i + 1, towers):
                distance = math.sqrt(
                    (tower_points[i][0] - tower_points[j][0]) ** 2 + (tower_points[i][1] - tower_points[j][1]) ** 2)
                if distance <= min(tower_radii[i], tower_radii[j]):
                    G.add_edge(i, j)

        is_connected = nx.is_connected(G)

        if is_connected:
            print("Attempt: %d - The graph G is connected." % att)
            return tower_points, tower_radii, G
        else:
            att = att+1

    print("The graph G is not connected.")
    exit(-1)


def create_regular_manhattan(towers, area_side):
    tower_points = []
    towers_per_side = math.isqrt(towers) - 1
    gap = int(area_side / towers_per_side)

    for x in range(0, area_side+1, gap):
        for y in range(0, area_side+1, gap):
            tower_points.append([x, y])

    tower_radii = []
    radius = gap
    for i in range(0, towers):
        tower_radii.append(radius)

    G = nx.Graph()
    for i in range(0, towers):
        G.add_node(i, pos=tower_points[i])
    for i in range(0, towers):
        for j in range(i + 1, towers):
            distance = math.sqrt(
                (tower_points[i][0] - tower_points[j][0]) ** 2 + (tower_points[i][1] - tower_points[j][1]) ** 2)
            if distance <= radius:
                G.add_edge(i, j)

    return tower_points, tower_radii, G


def create_regular_diagonal(towers, area_side):
    tower_points = []
    towers_per_side = math.isqrt(towers) - 1
    gap = int(area_side / towers_per_side)

    for x in range(0, area_side + 1, gap):
        for y in range(0, area_side + 1, gap):
            tower_points.append([x, y])

    tower_radii = []
    radius = gap*math.sqrt(2)
    for i in range(0, towers):
        tower_radii.append(radius)

    G = nx.Graph()
    for i in range(0, towers):
        G.add_node(i, pos=tower_points[i])
    for i in range(0, towers):
        for j in range(i + 1, towers):
            distance = math.sqrt(
                (tower_points[i][0] - tower_points[j][0]) ** 2 + (tower_points[i][1] - tower_points[j][1]) ** 2)
            if distance <= radius:
                G.add_edge(i, j)

    return tower_points, tower_radii, G


def create_ring_lattice(towers, lattice_neighbors, area_side):
    r = (2/3.) * area_side / 2
    center_x = area_side / 2
    center_y = area_side / 2

    angle_step = 2 * math.pi / towers

    tower_points = []

    for i in range(towers):
        angle = i * angle_step
        x = center_x + r * math.cos(angle)
        y = center_y + r * math.sin(angle)
        tower_points.append((x, y))

    side_length = 2 * r * math.sin(math.pi / towers) + 5
    radius = side_length * (lattice_neighbors / 2)

    tower_radii = []
    for i in range(0, towers):
        tower_radii.append(radius)

    G = nx.Graph()
    for i in range(0, towers):
        G.add_node(i, pos=tower_points[i])
    for i in range(0, towers):
        for j in range(i + 1, towers):
            distance = math.sqrt(
                (tower_points[i][0] - tower_points[j][0]) ** 2 + (tower_points[i][1] - tower_points[j][1]) ** 2)
            if distance <= radius:
                G.add_edge(i, j)

    return tower_points, tower_radii, G


def create_star(towers, star_edges, area_side):
    r = area_side / 2
    rp = r*2./3
    center_x = area_side / 2
    center_y = area_side / 2

    angle_step = 2 * math.pi / star_edges

    tower_points = []
    intermediate_points = []

    # Central tower
    tower_points.append((center_x, center_y))

    # Intermediate towers
    for i in range(star_edges):
        angle = i * angle_step
        x = center_x + rp * math.cos(angle)
        y = center_y + rp * math.sin(angle)
        tower_points.append((x, y))
        intermediate_points.append((x, y))

    radius = (2/3.) * rp * math.sin(angle_step/2.)

    # External towers
    for int_x, int_y in intermediate_points:
        for i in range(star_edges):
            angle = i * angle_step
            x = int_x + (radius) * math.cos(angle)
            y = int_y + (radius) * math.sin(angle)
            tower_points.append((x, y))

    tower_radii = []
    for i in range(0, towers):
        tower_radii.append(radius)

    G = nx.Graph()
    for i in range(0, towers):
        G.add_node(i, pos=tower_points[i])

    # Central - Intermediate
    for i in range(1, star_edges+1):
        G.add_edge(0, i)

    # Intermediate - External
    for i in range(1, star_edges + 1):
        for j in range(star_edges*i + 1, star_edges*(i+1) + 1):
            G.add_edge(i, j)

    return tower_points, tower_radii, G


def create_bus(towers, area_side):
    r = area_side / 2
    rp = r*2./3
    center_x = area_side / 2
    center_y = area_side / 2

    angle_step = 2 * math.pi / (towers-1)

    tower_points = []
    intermediate_points = []

    # Central tower
    tower_points.append((center_x, center_y))

    # Intermediate towers
    for i in range(towers-1):
        angle = i * angle_step
        x = center_x + rp * math.cos(angle)
        y = center_y + rp * math.sin(angle)
        tower_points.append((x, y))
        intermediate_points.append((x, y))

    radius = 2 * rp * math.sin(angle_step/2.)

    tower_radii = []
    for i in range(0, towers):
        tower_radii.append(radius)

    G = nx.Graph()
    for i in range(0, towers):
        G.add_node(i, pos=tower_points[i])

    # Central - Intermediate
    for i in range(1, towers):
        G.add_edge(0, i)

    return tower_points, tower_radii, G