import math

import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import math
from sympy import Polygon, Point

from util import get_distance

'''
Write here the code related to problem instances generation
'''


def generate_problem_instance(config):
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

    random.seed(seed)

    # Area
    points = [Point(0, 0), Point(area_side, 0), Point(area_side, area_side), Point(0, area_side), Point(0, 0)]
    area_x_coords = []
    area_y_coords = []
    for i in range(0, len(points)):
        area_x_coords.append(points[i].x)
        area_y_coords.append(points[i].y)

    # Towers
    tower_points = []
    tower_radii = []
    G = nx.Graph()

    if scenario == -1:
        tower_points, tower_radii, G = create_test(config)
    elif scenario == 0:
        print("Not implemented yet!")
        exit(1)
    elif scenario == 1:
        tower_points, tower_radii, G = create_RGG_fixed_radius(radius_min, towers, area_side)
    elif scenario == 2:
        tower_points, tower_radii, G = create_RGG_variable_radius(radius_min, radius_max, towers, area_side)
    elif scenario == 3:
        tower_points, tower_radii, G = create_regular_manhattan(towers, area_side)
    elif scenario == 4:
        tower_points, tower_radii, G = create_regular_diagonal(towers, area_side)
    elif scenario == 5:
        print("Not implemented yet!")
        exit(1)
    elif scenario == 6:
        tower_points, tower_radii, G = create_ring_lattice(towers, lattice_neighbors, area_side)

    # Trajectories
    trajectories_paths = []
    i = 0
    while i < trajectories:
        x_0 = random.uniform(0, area_side)
        y_0 = random.uniform(0, area_side)
        x_1 = random.uniform(0, area_side)
        y_1 = random.uniform(0, area_side)
        dist = math.sqrt((x_1 - x_0) ** 2 + (y_1 - y_0) ** 2)
        if dist < min_dist_trajectory:
            continue

        trajectories_paths.append([(x_0, y_0), (x_1, y_1)])
        i = i + 1

    if scenario == -1:
        trajectories_paths = []
        trajectories_paths.append([(120, 305), (975, 100)])
        trajectories_paths.append([(0, 449.999), (1000, 450)])

    # Intersections + Intervals
    intersection_points = []
    for path in trajectories_paths:
        x_0, y_0 = path[0]
        x_1, y_1 = path[1]

        p0 = sp.Point(x_0, y_0)
        p1 = sp.Point(x_1, y_1)

        intersections = []
        for i in range(0, towers):
            tower_x, tower_y = tower_points[i]
            radius = tower_radii[i]
            segment = sp.Segment(p0, p1)
            circle = sp.Circle(sp.Point(tower_x, tower_y), radius)
            ints = (segment.intersection(circle))
            if ints:
                if len(ints) == 2:
                    intersections.append((ints, i))
                else:
                    if p0.distance(circle.center) < circle.radius:
                        ints.append(p0)
                    elif p1.distance(circle.center) < circle.radius:
                        ints.append(p1)
                    else:
                        # Tangent case, empty interval, so skip
                        break

                    intersections.append((ints, i))

        intersection_points.append(intersections)

    # Plot
    plt.figure(figsize=(8, 8))
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
    x = [pos[node][0] for node in G.nodes()]
    y = [pos[node][1] for node in G.nodes()]
    plt.scatter(x, y, color='orange')
    for edge in G.edges():
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
        print("P%d = [0, %.2f]" % (i, dist))
        # print(" S=(%.2f, %.2f)" % (x_0, y_0))
        # print(" D=(%.2f, %.2f)" % (x_1, y_1))

        for j in range(0, len(intersections)):
            ints = intersections[j]
            tower_id = ints[1]
            x_values = []
            y_values = []
            # print("  I -> T%d" % tower_id)

            i0x = ints[0][0].x
            i0y = ints[0][0].y
            i1x = ints[0][1].x
            i1y = ints[0][1].y
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

            print("  T%d - [%.2f, %.2f]" % (tower_id, min(dist_i0, dist_i1), max(dist_i0, dist_i1)))

            plt.plot(x_values, y_values, color='blue', linestyle='dashed', linewidth=0.7)

        print()

    plt.show()


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

def create_RGG_fixed_radius(radius, towers, area_side):
    max_attempts = 100
    att = 0
    while att < max_attempts:
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
            return tower_points, tower_radii, G

    print("The graph G is not connected.")


def create_RGG_variable_radius(radius_min, radius_max, towers, area_side):
    max_attempts = 100
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
            return tower_points, tower_radii, G

    print("The graph G is not connected.")


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
    r = area_side / 2
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
