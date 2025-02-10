import math
import time
import problem_gen as problem
from experiments import run_experiments_paper, visualize_exp_paper, fix_exp_results, get_plots_aggregated, new_plots, run_experiments
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
def is_square(n):
    return math.isqrt(n) ** 2 == n


def main():
    parser = argparse.ArgumentParser(description="Run optimization experiments in sensor networks with a drone.")

    parser.add_argument("--mode", type=str, choices=["single", "all", "plots"], required=True,
                        help="Execution mode: 'single' for a single instance, 'all' for all experiments, 'plots' for visualization.")

    # Parameters for single instance execution
    parser.add_argument("--area_side", type=int, default=1000, help="Side length of the area in meters.")
    parser.add_argument("--scenario", type=int, choices=[-1, 1, 2, 3, 4, 5, 6, 7], default=1,
                        help="Scenario selection.")
    parser.add_argument("--towers", type=int, default=250, help="Number of sensor towers.")
    parser.add_argument("--radius_min", type=int, default=100, help="Minimum sensing radius.")
    parser.add_argument("--radius_max", type=int, default=300, help="Maximum sensing radius.")
    parser.add_argument("--lattice_neighbors", type=int, default=100, help="Lattice neighbors (even, >=2, <=TOWERS/2).")
    parser.add_argument("--star_edges", type=int, default=5, help="Number of star edges (>=3).")
    parser.add_argument("--trajectories", type=int, default=100, help="Number of drone trajectories.")
    parser.add_argument("--min_dist_trajectory", type=int, default=500, help="Minimum trajectory distance.")
    parser.add_argument("--algorithm", type=int, choices=range(7), default=6, help="Algorithm selection (0-6).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--iterations", type=int, default=25, help="Number of iterations.")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode.")

    args = parser.parse_args()

    print(f"Starting execution mode: {args.mode}")

    if args.mode == "single":
        print("Running a single experiment instance...")
        # Adjust parameters based on scenario requirements
        if args.scenario in [3, 4] and not is_square(args.towers):
            args.towers = (math.isqrt(args.towers) + 1) ** 2
        if args.scenario == 6:
            args.towers = max(args.towers, 3)
            args.lattice_neighbors = max(2, args.lattice_neighbors - (args.lattice_neighbors % 2))
        if args.scenario == 7:
            args.towers = args.star_edges ** 2 + args.star_edges + 1

        hyperparameters = {
            "area_side": args.area_side,
            "towers": args.towers,
            "radius_min": args.radius_min,
            "radius_max": args.radius_max,
            "trajectories": args.trajectories,
            "min_dist_trajectory": args.min_dist_trajectory,
            "scenario": args.scenario,
            "lattice_neighbors": args.lattice_neighbors,
            "star_edges": args.star_edges,
            "debug": args.debug,
        }
        print(f"Hyperparameters: {hyperparameters}")
        run_experiments(args.iterations, hyperparameters, args.algorithm)
        print("Single experiment execution completed.")

    elif args.mode == "all":
        print("Running all experiments...")
        scenarios = [1, 2, 3, 4, 6, 7]
        for scenario in scenarios:
            print(f"Running experiments for scenario {scenario}...")
            run_experiments_paper(scenario)
        print("All experiments execution completed.")

    elif args.mode == "plots":
        print("Generating plots...")
        visualize_exp_paper()
        fix_exp_results()
        get_plots_aggregated()
        new_plots()
        print("Plot generation completed.")


if __name__ == "__main__":
    main()


