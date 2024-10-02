import math
import time
import problem_gen as problem
from algorithms import *
from util import is_square

######################################### HYPER-PARAMETERS #############################################################

# In meters
AREA_SIDE = 1000

# There are many scenarios:
# -1 - Test (fixed)

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

TOWERS = 250

RADIUS_MIN = 100

# it must be >= RADIUS_MIN
RADIUS_MAX = 300

# it must be >= 2, even, and <= TOWERS/2
LATTICE_NEIGHBORS = 100

# it must be >= 3
STAR_EDGES = 5

TRAJECTORIES = 100

# In meters, must be less than AREA_SIDE*sqrt(2)
MIN_DIST_TRAJECTORY = 500

# 0 - MEP   alg_E_MEP
# 1 - MEP-k [not used-implemented]
# 2 - MTCP  alg_C_MTCP
# 3 - MEPT  alg_OPT_MEPT
# 4 - MEPT  alg_E_SC_MEPT
# 5 - MEPT  alg_E_T_MEPT
# ---------
# 6 - MEPT v1, v2 vs OPT
ALGORITHM = 6

SEED = 0

ITERATIONS = 25

DEBUG = False
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
        start_time = time.time()
        instance = problem.generate_problem_instance(config)
        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"generate_problem_instance execution time: {round(elapsed_time, 4)} s")

        # Algorithms
        if ALGORITHM == 0:
            # Minimum Eccentricity Problem - MEP
            output = alg_E_MEP(instance)
            print(output)
        # elif ALGORITHM == 1:
        #     # MEP-k
        #     output = single_minimum_k_coverage(instance)
        #     print(output)
        elif ALGORITHM == 2:
            # Minimum Tower Coverage Problem - MTCP
            output = alg_C_MTCP(instance)
            print(output)
        elif ALGORITHM == 3:
            # Minimum Eccentricity Problem with multiple Trajectories - MEPT
            output = alg_OPT_MEPT(instance)
            print(output)
        elif ALGORITHM == 4:
            # Minimum Eccentricity Problem with multiple Trajectories - MEPT
            output = alg_E_SC_MEPT(instance)
            print(output)
        elif ALGORITHM == 5:
            # Minimum Eccentricity Problem with multiple Trajectories - MEPT
            output = alg_E_T_MEPT(instance)
            print(output)
        elif ALGORITHM == 6:
            output_v1 = alg_E_SC_MEPT(instance)
            print(output_v1)
            output_v2 = alg_E_T_MEPT(instance)
            print(output_v2)
            output_opt = alg_OPT_MEPT(instance)
            print(output_opt)
            print("########################")
