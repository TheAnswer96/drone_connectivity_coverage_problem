import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

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

    # 1 - generate a square of size "area_side" x "area_side"
    area = [(0, 0), (area_side, 0), (area_side, area_side), (0, area_side)]
    plt.plot([x for x, y in area], [y for x, y in area], 'k-')

    # 2 - randomly generate "towers" points with x,y coordinates each within 0 and "area_side"
    random.seed(seed)
    tower_points = [(random.uniform(0, area_side), random.uniform(0, area_side)) for _ in range(towers)]


    # 3 - for each tower, generate randomly a radius within "radius_min" and "radius_max"
    tower_radii = [random.uniform(radius_min, radius_max) for _ in range(towers)]


    # 4 - generate "trajectories" trajectories by randomizing two endpoints each, each within 0 and "area_side"
    trajectories_path = [[(random.uniform(0, area_side), random.uniform(0, area_side)) for _ in range(2)] for _ in
                           range(trajectories)]

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
