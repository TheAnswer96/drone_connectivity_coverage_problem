import networkx as nx
import random
from queue import Queue
import problem_gen as problem


######################################### HYPER-PARAMETERS #############################################################
AREA_SIDE = 1000  # [m], it is squared
TOWERS = 2 # number of towers
RADIUS_MIN = 180  # [m], for the connectivity
RADIUS_MAX = 350 # it must be >= RADIUS_MIN
TRAJECTORIES = 2  # number of trajectories

SEED = 2
DEBUG = True
########################################################################################################################

if __name__ == '__main__':
    config = {
        "area_side": AREA_SIDE,
        "towers": TOWERS,
        "radius_min": RADIUS_MIN,
        "radius_max": RADIUS_MAX,
        "trajectories": TRAJECTORIES,
        "seed": SEED,
        "debug": DEBUG
    }

    problem.generate_problem_instance(config)
