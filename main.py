import math

import problem_gen as problem
from algorithms import *
from util import is_square

######################################### HYPER-PARAMETERS #############################################################

# In meters
AREA_SIDE = 1000

# There are many scenarios:
# -1 - Test (fixed)

#  0 - Complete graph
#      ???? (set cover problem, trivial)

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

#  5 - Bus
#      ignores both RADIUS_MIN and RADIUS_MAX
#      is connected
#      can't be fully covered

#  6 - Ring Lattice
#      ignores both RADIUS_MIN and RADIUS_MAX
#      TOWERS must be >= 3 (if not, it is adjusted to 3)
#      is connected
#      can't be fully covered

#  7 - Star
#      ignores both RADIUS_MIN and RADIUS_MAX and TOWERS
#      TOWERS will be set to STAR_EDGES^2 + STAR_EDGES + 1
#      is connected
#      can't be fully covered
SCENARIO = 1

TOWERS = 50

RADIUS_MIN = 170

# it must be >= RADIUS_MIN
RADIUS_MAX = 300

# it must be >= 2, even, and <= TOWERS/2
LATTICE_NEIGHBORS = 4

# it must be >= 3
STAR_EDGES = 5

TRAJECTORIES = 2

# In meters, must be less than AREA_SIDE*sqrt(2)
MIN_DIST_TRAJECTORY = 500

ALGORITHM = 5

SEED = 0

ITERATIONS = 25

DEBUG = True
########################################################################################################################

if __name__ == '__main__':

    if SCENARIO == 3 or SCENARIO == 4:
        if not is_square(TOWERS):
            TOWERS = (math.isqrt(TOWERS) + 1) ** 2

    if SCENARIO == 6:
        if TOWERS < 3:
            TOWERS = 3

        if LATTICE_NEIGHBORS % 2 == 1:
            LATTICE_NEIGHBORS = LATTICE_NEIGHBORS - 1

        if LATTICE_NEIGHBORS < 2:
            LATTICE_NEIGHBORS = 2

        # if LATTICE_NEIGHBORS > TOWERS / 2:
        #     LATTICE_NEIGHBORS = TOWERS / 2

    if SCENARIO == 7:
        TOWERS = STAR_EDGES**2 + STAR_EDGES + 1

    for i in range(1, ITERATIONS+1):
        print(f"Iteration {i}/{ITERATIONS}")
        # Input parameters
        config = {
            "area_side": AREA_SIDE,
            "towers": TOWERS,
            "radius_min": RADIUS_MIN,
            "radius_max": RADIUS_MAX,
            "trajectories": TRAJECTORIES,
            "min_dist_trajectory": MIN_DIST_TRAJECTORY,
            "scenario": SCENARIO,
            "lattice_neighbors": LATTICE_NEIGHBORS,
            "star_edges": STAR_EDGES,
            "seed": i,
            "debug": DEBUG
        }

        # Random instance
        instance = problem.generate_problem_instance(config)

        # Algorithms
        output = []
        if ALGORITHM == 0:
            # MEP
            output = single_minimum_eccentricity(instance)
        elif ALGORITHM == 1:
            # MTCP
            output = single_minimum_coverage(instance)
        elif ALGORITHM == 2:
            # MEP-k
            output = single_minimum_k_coverage(instance)
        elif ALGORITHM == 3:
            # MEPT
            output = multiple_minimum_eccentricity_opt(instance)
        elif ALGORITHM == 4:
            # MEPT
            output = multiple_minimum_eccentricity_v1(instance)
        elif ALGORITHM == 5:
            # MEPT
            output_v2 = multiple_minimum_eccentricity_v2(instance)
            print(output_v2)
            output_opt = multiple_minimum_eccentricity_opt(instance)
            print(output_opt)
            if output_opt["total_towers"] > output_v2["total_towers"]:
                print(f"Error - seed={i}")
            print("########################")



