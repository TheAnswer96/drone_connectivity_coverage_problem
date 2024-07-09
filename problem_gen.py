import math

import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import math
from sympy import Polygon, Point

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
    seed = config["seed"]
    debug = config["debug"]

    random.seed(seed)

    # generate a square of size "area_side" x "area_side"
    points = [Point(0, 0), Point(area_side, 0), Point(area_side, area_side), Point(0, area_side), Point(0, 0)]
    area_x_coords = []
    area_y_coords = []
    for i in range(0, len(points)):
        area_x_coords.append(points[i].x)
        area_y_coords.append(points[i].y)

    # generate towers
    tower_points = []
    tower_radii = []
    G = nx.Graph()

    if scenario == 0:
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

    # 4 - generate "trajectories" trajectories by randomizing two endpoints each, each within 0 and "area_side"
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

    # 5 - for each trajectory:
    # 5.1 - compute the intersection points among it and all the towers

    # intersection_points = []
    # for path in trajectories_paths:
    #     x_0, y_0 = path[0]
    #     x_1, y_1 = path[1]
    #
    #     intersections = []
    #     for i in range(0, towers):
    #         tower_x, tower_y = tower_points[i]
    #         radius = tower_radii[i]
    #         segment = sp.Segment(sp.Point(x_0, y_0), sp.Point(x_1, y_1))
    #         circle = sp.Circle(sp.Point(tower_x, tower_y), radius)
    #         ints = (segment.intersection(circle))
    #         if ints:
    #             intersections.append((ints, tower_points[i]))
    #
    #     intersection_points.append(intersections)

    # 5.2 - create the intervals. This is difficult. You can imagine this interval as a segment
    #       that goes from 0 (observer) to a certain distance (other endpoint).
    #       From 0, you compute the distance of all the intersection points, and then you can associate the intervals.

    # 6 - build the networkx graph from what you have done before
    # 6.1 - generate "towers" vertices, and assign the coordinates you created before
    # 6.2 - add an edge between vertex v_i and v_j if their Euclidean distance is within the min{r_i, r_j}

    # 0 - IMPORTANT: draw what you are doing. Use matplot lib to draw exactly every previous step I listed.
    #     It will help you to see if you are doing well or not, and also us to guide you in case you need assistance

    plt.figure(figsize=(8, 8))

    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Area')
    plt.grid(True)

    plt.plot(area_x_coords, area_y_coords, marker='o')
    plt.fill(area_x_coords, area_y_coords, alpha=0.05)

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

    # # Plot the Observer and Destination for each trajectory
    # for i in range(0, len(trajectories_paths)):
    #     x_0, y_0 = trajectories_paths[i][0]
    #     x_1, y_1 = trajectories_paths[i][1]
    #     plt.scatter(x_0, y_0, marker='*', color='red', label='Observer')  # Observer
    #     plt.scatter(x_1, y_1, marker='D', color='green', label='Destination')  # Destination

    # for i in range(0, len(trajectories_paths)):
    #     x_values = [trajectories_paths[i][0][0], trajectories_paths[i][1][0]]
    #     y_values = [trajectories_paths[i][0][1], trajectories_paths[i][1][1]]
    #     plt.text(trajectories_paths[i][0][0] + 10, trajectories_paths[i][0][1] + 10, 'P' + str(i), fontsize=12)
    #     plt.plot(x_values, y_values, marker='o')

    # for intersections in intersection_points:
    #     for ints in intersections:
    #         tower = ints[1]
    #         x_values = []
    #         y_values = []
    #         x_values.append(tower[0])
    #         y_values.append(tower[1])
    #         for loc_int in ints[0]:
    #             x_values.append(loc_int.x)
    #             y_values.append(loc_int.y)
    #             plt.plot(loc_int.x, loc_int.y, marker='x')
    #
    #         x_values.append(tower[0])
    #         y_values.append(tower[1])
    #         plt.plot(x_values, y_values, color='black', linestyle='dashed')

    plt.show()


def create_RGG_fixed_radius(radius, towers, area_side):
    # randomly generate "towers" points with x,y coordinates each within 0 and "area_side"
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

    return tower_points, tower_radii, G


def create_RGG_variable_radius(radius_min, radius_max, towers, area_side):
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

    return tower_points, tower_radii, G


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
