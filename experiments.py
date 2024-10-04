import os
import time

import problem_gen as problem
from algorithms import *


def run_experiments(iterations, hyperparams, algorithm):
    if not os.path.exists("./exp"):
        print("exp directory creation.")
        os.makedirs("./exp")

    for i in range(1, iterations + 1):
        print(f"Iteration {i}/{iterations}")
        # Input parameters
        config = {
            "area_side": hyperparams["area_side"],
            "towers": hyperparams["towers"],
            "radius_min": hyperparams["radius_min"],
            "radius_max": hyperparams["radius_max"],
            "trajectories": hyperparams["trajectories"],
            "min_dist_trajectory": hyperparams["min_dist_trajectory"],
            "scenario": hyperparams["scenario"],
            "lattice_neighbors": hyperparams["lattice_neighbors"],
            "star_edges": hyperparams["star_edges"],
            "seed": i,
            "debug": hyperparams["debug"]
        }

        # Random instance
        start_time = time.time()
        instance = problem.generate_problem_instance(config)
        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"generate_problem_instance execution time: {round(elapsed_time, 4)} s")

        # Algorithms
        if algorithm == 0:
            # Minimum Eccentricity Problem - MEP
            output = alg_E_MEP(instance)
            print(output)
            # elif ALGORITHM == 1:
            #     # MEP-k
            #     output = single_minimum_k_coverage(instance)
            #     print(output)
            exit(1)
        elif algorithm == 2:
            # Minimum Tower Coverage Problem - MTCP
            output = alg_C_MTCP(instance)
            print(output)
        elif algorithm == 3:
            # Minimum Eccentricity Problem with multiple Trajectories - MEPT
            output = alg_OPT_MEPT(instance)
            print(output)
        elif algorithm == 4:
            # Minimum Eccentricity Problem with multiple Trajectories - MEPT
            output = alg_E_SC_MEPT(instance)
            print(output)
        elif algorithm == 5:
            # Minimum Eccentricity Problem with multiple Trajectories - MEPT
            output = alg_E_T_MEPT(instance)
            print(output)
        elif algorithm == 6:
            output_v1 = alg_E_SC_MEPT(instance)
            print(output_v1)
            output_v2 = alg_E_T_MEPT(instance)
            print(output_v2)
            output_opt = alg_OPT_MEPT(instance)
            print(output_opt)
            print("########################")
    return
