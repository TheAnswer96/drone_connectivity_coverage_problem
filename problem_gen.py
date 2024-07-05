import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
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
    seed = config["seed"]
    debug = config["debug"]

    random.seed(seed)

    # 1 - generate a square of size "area_side" x "area_side"
    points = [Point(0, 0), Point(area_side, 0), Point(area_side, area_side), Point(0, area_side), Point(0, 0)]
    area_x_coords = []
    area_y_coords = []
    for i in range(0, len(points)):
        area_x_coords.append(points[i].x)
        area_y_coords.append(points[i].y)

    # 2 - randomly generate "towers" points with x,y coordinates each within 0 and "area_side"
    tower_points = []
    for i in range(0, towers):
        x = random.uniform(0, area_side)
        y = random.uniform(0, area_side)
        tower_points.append([x, y])

    # 3 - for each tower, generate randomly a radius within "radius_min" and "radius_max"
    tower_radii = []
    for i in range(0, towers):
        radius = random.uniform(radius_min, radius_max)
        tower_radii.append(radius)

    # 4 - generate "trajectories" trajectories by randomizing two endpoints each, each within 0 and "area_side"
    trajectories_paths = []
    for i in range(0, trajectories):
        x_1 = random.uniform(0, area_side)
        y_1 = random.uniform(0, area_side)
        x_2 = random.uniform(0, area_side)
        y_2 = random.uniform(0, area_side)
        trajectories_paths.append([(x_1, y_1), (x_2, y_2)])


    # 5 - for each trajectory:
    # 5.1 - compute the intersection points among it and all the towers
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

    for i in range(0, towers):
        tower_x = tower_points[i][0]
        tower_y = tower_points[i][1]
        radius = tower_radii[i]

        plt.scatter(tower_x, tower_y, marker='o', color='orange')
        circle = plt.Circle((tower_x, tower_y), radius, color='orange', alpha=0.1)
        plt.gca().add_patch(circle)

    for path in trajectories_paths:
        x_values = [path[0][0], path[1][0]]
        y_values = [path[0][1], path[1][1]]
        plt.plot(x_values, y_values, marker='o')

    plt.show()





