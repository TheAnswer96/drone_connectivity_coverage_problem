import os
import time

import problem_gen as problem
from algorithms import *
import pandas as pd

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

dir_dict = {
    1: "rgg_fixed",
    2: "rgg_variable",
    3: "manhattan",
    4: "diagonal",
    5: "bus",
    6: "lattice",
    7: "star",
}

# def run_experiments(iterations, hyperparams, algorithm):
#     if not os.path.exists("./exp"):
#         print("exp directory creation.")
#         os.makedirs("./exp")
#
#     for i in range(1, iterations + 1):
#         print(f"Iteration {i}/{iterations}")
#         # Input parameters
#         config = {
#             "area_side": hyperparams["area_side"],
#             "towers": hyperparams["towers"],
#             "radius_min": hyperparams["radius_min"],
#             "radius_max": hyperparams["radius_max"],
#             "trajectories": hyperparams["trajectories"],
#             "min_dist_trajectory": hyperparams["min_dist_trajectory"],
#             "scenario": hyperparams["scenario"],
#             "lattice_neighbors": hyperparams["lattice_neighbors"],
#             "star_edges": hyperparams["star_edges"],
#             "seed": i,
#             "debug": hyperparams["debug"]
#         }
#
#         # Random instance
#         start_time = time.time()
#         instance = problem.generate_problem_instance(config)
#         end_time = time.time()
#
#         elapsed_time = end_time - start_time
#         print(f"generate_problem_instance execution time: {round(elapsed_time, 4)} s")
#
#         # Algorithms
#         if algorithm == 0:
#             # Minimum Eccentricity Problem - MEP
#             output = alg_E_MEP(instance)
#             print(output)
#             # elif ALGORITHM == 1:
#             #     # MEP-k
#             #     output = single_minimum_k_coverage(instance)
#             #     print(output)
#             exit(1)
#         elif algorithm == 2:
#             # Minimum Tower Coverage Problem - MTCP
#             output = alg_C_MTCP(instance)
#             print(output)
#         elif algorithm == 3:
#             # Minimum Eccentricity Problem with multiple Trajectories - MEPT
#             output = alg_OPT_MEPT(instance)
#             print(output)
#         elif algorithm == 4:
#             # Minimum Eccentricity Problem with multiple Trajectories - MEPT
#             output = alg_E_SC_MEPT(instance)
#             print(output)
#         elif algorithm == 5:
#             # Minimum Eccentricity Problem with multiple Trajectories - MEPT
#             output = alg_E_T_MEPT(instance)
#             print(output)
#         elif algorithm == 6:
#             output_v1 = alg_E_SC_MEPT(instance)
#             print(output_v1)
#             output_v2 = alg_E_T_MEPT(instance)
#             print(output_v2)
#             output_opt = alg_OPT_MEPT(instance)
#             print(output_opt)
#             print("########################")
#     return

def run_experiments_paper():
    #### Parameters
    area_sides = [500]
    towers = [50, 100, 200, 350]
    scenarios = [2]
    radius_min = [100]
    radius_max = [300]
    lattice_neighbors = [100]
    star_edges = [5]
    trajectories = [10, 20, 50, 100]
    min_dist_trajectory = [500]
    iterations = 33
    debug = False

    if not os.path.exists("./exp"):
        print("exp directory creation.")
        os.makedirs("./exp")

    for scene in scenarios:
        pth = "./exp/"+dir_dict[scene]
        if not os.path.exists(pth):
            print(pth, " directory creation.")
            os.makedirs(pth)

    for area in area_sides:
        for tower in towers:
            for scenario in scenarios:
                if scenario !=6:
                    print("dummy lattice settled!")
                    lattice_neighbors = [500]
                if scenario != 7:
                    print("dummy star settled!")
                    star_edges = [5]
                for r_min in radius_min:
                    for r_max in radius_max:
                        for trj in trajectories:
                            for min_dist in min_dist_trajectory:
                                for lattice in lattice_neighbors:
                                    for star in star_edges:

                                        #             alg_E_SC_MEPT(instance)
                                        #             alg_E_T_MEPT(instance)
                                        #             alg_OPT_MEPT(instance)

                                        for min_traj in min_dist_trajectory:
                                            results = pd.DataFrame(
                                                columns=["iteration_seed", "time_opt", "eccentricity_opt",
                                                         "total_towers_opt", "time_e_sc_mept", "eccentricity_e_sc_mept",
                                                         "total_towers_e_sc_mept",
                                                         "time_e_t_mept", "time_e_t_mept", "eccentricity_e_t_mept",
                                                         "total_towers_e_t_mept"])

                                            destination = "./exp/"+dir_dict[scenario]+"/results_ar"+str(area)+"_t"+\
                                                          str(tower)+"_rn" + str(r_min) + "_rx"+str(r_max) + "_tr" +\
                                                        str(trj) + "_di"+str(min_traj) + "_l"+str(lattice)+"_s"+str(star) + ".csv"
                                            for i in range(1, iterations):
                                                tower, lattice, star = problem.preprocessing_scenario(scenario, tower, lattice, star)
                                                print(f"Iteration {i}/{iterations}")
                                                config = {
                                                            "area_side": area,
                                                            "towers": tower,
                                                            "radius_min": r_min,
                                                            "radius_max": r_max,
                                                            "trajectories": trj,
                                                            "min_dist_trajectory": min_traj,
                                                            "scenario": scenario,
                                                            "lattice_neighbors": lattice,
                                                            "star_edges": star,
                                                            "seed": i,
                                                            "debug": debug
                                                        }

                                                # Random instance
                                                start_time = time.time()
                                                instance = problem.generate_problem_instance(config)
                                                end_time = time.time()

                                                elapsed_time = end_time - start_time
                                                print(f"generate_problem_instance execution time: {round(elapsed_time, 4)} s")
                                                output_opt = alg_OPT_MEPT(instance)
                                                output_v1 = alg_E_SC_MEPT(instance)
                                                output_v2 = alg_E_T_MEPT(instance)
                                                result_row = [i, output_opt["elapsed_time"], output_opt["eccentricity"], output_opt["total_towers"],
                                                              output_v1["elapsed_time"], output_v1["eccentricity"], output_v1["total_towers"],
                                                              output_v2["elapsed_time"], output_v2["eccentricity"], output_v2["total_towers"],
                                                              ]
                                                results.loc[len(results)] = result_row
                                            results.to_csv(destination)
    return
