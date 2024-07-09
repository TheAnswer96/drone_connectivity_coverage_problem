import math

import networkx as nx
import random
from queue import Queue
import problem_gen as problem
from util import is_square

######################################### HYPER-PARAMETERS #############################################################

# In meters
AREA_SIDE = 1000

# There are many scenarios:
# -1 - Test (fixed)

#  0 - Complete graph
#      ????

#  1 - RGG with fixed r
#      considers only RADIUS_MIN
#      can be disconnected
#      can't be fully covered

#  2 - RGG with different r
#      considers both RADIUS_MIN and RADIUS_MAX
#      can be disconnected
#      can't be fully covered

#  3 - Regular Manhattan
#      ignores both RADIUS_MIN and RADIUS_MAX
#      TOWERS must be a square (if not, it is adjusted to next square)
#      is connected
#      fully covered

#  4 - Regular Diagonal
#      ignores both RADIUS_MIN and RADIUS_MAX
#      TOWERS must be a square (if not, it is adjusted to next square)
#      is connected
#      fully covered

#  5 - Path
#      ????

#  6 - Ring Lattice
#      ignores both RADIUS_MIN and RADIUS_MAX
#      TOWERS must be >= 3 (if not, it is adjusted to 3)
#      is connected
#      can't be fully covered
SCENARIO = 1

TOWERS = 5

RADIUS_MIN = 300

# it must be >= RADIUS_MIN
RADIUS_MAX = 350

# it must be >= 2, even, and <= TOWERS/2
LATTICE_NEIGHBORS = 4

TRAJECTORIES = 2

# In meters, must be less than AREA_SIDE*sqrt(2)
MIN_DIST_TRAJECTORY = 700

SEED = 10

DEBUG = True
########################################################################################################################

if __name__ == '__main__':

    if SCENARIO == 3 or SCENARIO == 4:
        if not is_square(TOWERS):
            TOWERS = (math.isqrt(TOWERS) + 1)**2

    if SCENARIO == 6:
        if TOWERS < 3:
            TOWERS = 3

        if LATTICE_NEIGHBORS % 2 == 1:
            LATTICE_NEIGHBORS = LATTICE_NEIGHBORS - 1

        if LATTICE_NEIGHBORS < 2:
            LATTICE_NEIGHBORS = 2

        # if LATTICE_NEIGHBORS > TOWERS / 2:
        #     LATTICE_NEIGHBORS = TOWERS / 2

    config = {
        "area_side": AREA_SIDE,
        "towers": TOWERS,
        "radius_min": RADIUS_MIN,
        "radius_max": RADIUS_MAX,
        "trajectories": TRAJECTORIES,
        "min_dist_trajectory": MIN_DIST_TRAJECTORY,
        "scenario": SCENARIO,
        "lattice_neighbors": LATTICE_NEIGHBORS,
        "seed": SEED,
        "debug": DEBUG
    }

    problem.generate_problem_instance(config)
