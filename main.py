import math
import time
import problem_gen as problem
from experiments import run_experiments_paper, visualize_exp_paper, fix_exp_results
from util import is_square
import argparse

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
    # parser = argparse.ArgumentParser(description="Initialize experiments...")
    #
    # parser.add_argument("--area_side", type=int, default=1000, help="The side of the area you want to use. Express it in meters. [Default=1000]")
    # parser.add_argument("--scenario", type=int, default=1, help="Scenario number ranges from -1 to 7. [Default=1]")
    # parser.add_argument("--towers", type=int, default=250, help="Number of towers. [Default=250].")
    # parser.add_argument("--radius_min", type=int, default=100, help="Minimum radius in meters. [Default=100]")
    # parser.add_argument("--radius_max", type=int, default=300,
    #                     help="Maximum radius in meters, must be >= RADIUS_MIN. [Default=300]")
    # parser.add_argument("--lattice_neighbors", type=int, default=100,
    #                     help="Must be >= 2, even, and <= TOWERS/2. [Default=100]")
    # parser.add_argument("--star_edges", type=int, default=5, help="Number of star edges, must be >= 3. [Default=5]")
    # parser.add_argument("--trajectories", type=int, default=100, help="Number of trajectories. [Default=100]")
    # parser.add_argument("--min_dist_trajectory", type=int, default=500,
    #                     help="Minimum distance of trajectory in meters, must be less than AREA_SIDE * sqrt(2). [Default=500]")
    # parser.add_argument("--algorithm", type=int, default=6, help="Algorithm selection from 0 to 6. [Default=6]")
    # parser.add_argument("--seed", type=int, default=0, help="Random seed. [Default=0]")
    # parser.add_argument("--iterations", type=int, default=25, help="Number of iterations. [Default=25]")
    # parser.add_argument("--debug", action='store_true', help="Enable debug mode. [Default=False]")
    #
    # args = parser.parse_args()
    #
    # AREA_SIDE = args.area_side
    # SCENARIO = args.scenario
    # TOWERS = args.towers
    # RADIUS_MIN = args.radius_min
    # RADIUS_MAX = args.radius_max
    # LATTICE_NEIGHBORS = args.lattice_neighbors
    # STAR_EDGES = args.star_edges
    # TRAJECTORIES = args.trajectories
    # MIN_DIST_TRAJECTORY = args.min_dist_trajectory
    # ALGORITHM = args.algorithm
    # SEED = args.seed
    # ITERATIONS = args.iterations
    # DEBUG = args.debug
    #
    # if SCENARIO == 3 or SCENARIO == 4:
    #     if not is_square(TOWERS):
    #         TOWERS = (math.isqrt(TOWERS) + 1) ** 2
    #
    # if SCENARIO == 6:
    #     if TOWERS < 3:
    #         TOWERS = 3
    #
    #     if LATTICE_NEIGHBORS % 2 == 1:
    #         LATTICE_NEIGHBORS = LATTICE_NEIGHBORS - 1
    #
    #     if LATTICE_NEIGHBORS < 2:
    #         LATTICE_NEIGHBORS = 2
    #
    #     # if LATTICE_NEIGHBORS > TOWERS / 2:
    #     #     LATTICE_NEIGHBORS = TOWERS / 2
    #
    # if SCENARIO == 7:
    #     TOWERS = STAR_EDGES**2 + STAR_EDGES + 1
    #
    # hyper = {
    #         "area_side": AREA_SIDE,
    #         "towers": TOWERS,
    #         "radius_min": RADIUS_MIN,
    #         "radius_max": RADIUS_MAX,
    #         "trajectories": TRAJECTORIES,
    #         "min_dist_trajectory": MIN_DIST_TRAJECTORY,
    #         "scenario": SCENARIO,
    #         "lattice_neighbors": LATTICE_NEIGHBORS,
    #         "star_edges": STAR_EDGES,
    #         "debug": DEBUG
    #     }
    #
    # run_experiments(ITERATIONS, hyper, ALGORITHM)

    #Run this function for exhaustive tests

    # scenarios = [1,2,3,4,6,7]
    # for scenario in scenarios:
    #     run_experiments_paper(scenario)


    visualize_exp_paper()
    # fix_exp_results()