import math

import networkx as nx
import random
from queue import Queue
import problem_gen as problem
from util import is_square

######################################### HYPER-PARAMETERS #############################################################

# In meters
AREA_SIDE = 1000

TOWERS = 36

# There are many scenarios:
# -1 - Test (fixed)

#  0 - Complete graph ????

#  1 - RGG with fixed r
#      considers only RADIUS_MIN
#      can be disconnected

#  2 - RGG with different r
#      considers both RADIUS_MIN and RADIUS_MAX
#      can be disconnected

#  3 - Regular Manhattan
#      ignores both RADIUS_MIN and RADIUS_MAX
#      TOWERS must be a square (in negative case, it is adjusted to next square)
#      is connected

#  4 - Regular Diagonal
#      ignores both RADIUS_MIN and RADIUS_MAX
#      TOWERS must be a square (in negative case, it is adjusted to next square)
#      is connected

#  5 - Path
#  6 - Ring Lattice
SCENARIO = 3

RADIUS_MIN = 50

# it must be >= RADIUS_MIN
RADIUS_MAX = 350

TRAJECTORIES = 1

# In meters, must be less than AREA_SIDE*sqrt(2)
MIN_DIST_TRAJECTORY = 700

SEED = 223
DEBUG = True
########################################################################################################################

if __name__ == '__main__':

    if not is_square(TOWERS):
        TOWERS = (math.isqrt(TOWERS) + 1)**2

    config = {
        "area_side": AREA_SIDE,
        "towers": TOWERS,
        "radius_min": RADIUS_MIN,
        "radius_max": RADIUS_MAX,
        "trajectories": TRAJECTORIES,
        "min_dist_trajectory": MIN_DIST_TRAJECTORY,
        "scenario": SCENARIO,
        "seed": SEED,
        "debug": DEBUG
    }

    problem.generate_problem_instance(config)
