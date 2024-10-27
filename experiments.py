import os
import time
import pandas as pd

import problem_gen as problem
from algorithms import *
from util import get_exp_name, plot_aggregate, get_confidence
import scipy.stats as st
import numpy as np

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


def run_experiments_paper(scene):
    if not os.path.exists("./exp"):
        print("exp directory creation.")
        os.makedirs("./exp")

    pth = "./exp/"+dir_dict[scene]
    if not os.path.exists(pth):
        print(pth, " directory creation.")
        os.makedirs(pth)

    #### Parameters
    area_sides = [1000]
    min_dist_trajectory = [int(area/3*2) for area in area_sides]  # 2/3 area side

    trajectories = [1, 10, 20, 50, 100]

    iterations = 33
    debug = False

    if scene == 1:
        towers = [50, 100, 200, 350]
        radius = [100, 150, 200, 250, 300]
        run_RGG_fixed(area_sides, towers, radius, trajectories, min_dist_trajectory, iterations, dir_dict, debug)
    elif scene == 2:
        towers = [50, 100, 200, 350]
        radius = [(100, 300), (150, 300), (200, 300), (250, 300)]
        run_RGG_variable(area_sides, towers, radius, trajectories, min_dist_trajectory, iterations, dir_dict, debug)
    elif scene == 3:
        towers = [4**2, 6**2, 7**2, 8**2, 10**2, 12**2, 14**2, 15**2, 16**2]
        run_regular_manhattan(area_sides, towers, trajectories, min_dist_trajectory, iterations, dir_dict, debug)
    elif scene == 4:
        towers = [4**2, 6**2, 7**2, 8**2, 10**2, 12**2, 14**2, 15**2, 16**2]
        run_regular_diagonal(area_sides, towers, trajectories, min_dist_trajectory, iterations, dir_dict, debug)
    elif scene == 5:
        raise "not yet implemented!"
    elif scene == 6:
        towers = [5, 10, 15, 20]
        lattice_neighbors = [2, 4, 6]
        run_lattice(area_sides, towers, lattice_neighbors, trajectories, min_dist_trajectory, iterations, dir_dict, debug)
    elif scene == 7:
        star_edges = [5, 7, 10, 12]
        towers = [-1]
        run_star(area_sides, towers, star_edges, trajectories, min_dist_trajectory, iterations, dir_dict, debug)

    return


def run_RGG_fixed(areas, towers, rads, n_traj, traj_sizes, iterations, dict_sc, debug):
    print(f"RGG with fixed radius started...")
    for area in areas:
        for tower in towers:
            for rad in rads:
                for n in n_traj:
                    for size in traj_sizes:
                        if (tower < 100 and rad < 200) or (tower < 150 and rad < 150):
                            continue

                        print(f"exp area {area}, towers {tower}, rad {rad}, n traj {n}, min traj size {size}.")
                        output = pd.DataFrame(
                            columns=["iteration_seed", "time_opt", "eccentricity_opt", "total_towers_opt",
                                     "time_e_sc_mept", "eccentricity_e_sc_mept", "total_towers_e_sc_mept",
                                     "time_e_t_mept", "eccentricity_e_t_mept", "total_towers_e_t_mept"])

                        # creazione path di salvataggio
                        destination = get_exp_name(1, rad, 0, tower, area, 0, 0, n, size, dict_sc)
                        for i in range(1, iterations+1):
                            tower, lattice, star = problem.preprocessing_scenario(1, tower, 0, 0)

                            print(f"Iteration {i}/{iterations}")
                            config = {
                                "area_side": area,
                                "towers": tower,
                                "radius_min": rad,
                                "radius_max": 0,
                                "trajectories": n,
                                "min_dist_trajectory": size,
                                "scenario": 1,
                                "lattice_neighbors": 0,
                                "star_edges": 0,
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
                            print(
                                f"solution retrieving time OPT: {round(output_opt['elapsed_time'], 4)} s, MPT_SC: {round(output_v1['elapsed_time'], 4)}, MEPT_T: {round(output_v2['elapsed_time'], 4)}")
                            result_row = [i, output_opt["elapsed_time"], output_opt["eccentricity"],
                                          output_opt["total_towers"],
                                          output_v1["elapsed_time"], output_v1["eccentricity"],
                                          output_v1["total_towers"],
                                          output_v2["elapsed_time"], output_v2["eccentricity"],
                                          output_v2["total_towers"],
                                          ]
                            output.loc[len(output)] = result_row
                        output.to_csv(destination)
    print(f"RGG with fixed radius completed.")
    return


def run_RGG_variable(areas, towers, rads, n_traj, traj_sizes, iterations, dict_sc, debug):
    print(f"RGG with variable radius stated...")
    for area in areas:
        for tower in towers:
            for min_rad, max_rad in rads:
                for n in n_traj:
                    for size in traj_sizes:
                        if tower < 100 and min_rad < 200:
                            continue

                        print(f"exp area {area}, towers {tower}, rad ({min_rad},{max_rad}), n traj {n}, min traj size {size}.")
                        output = pd.DataFrame(
                            columns=["iteration_seed", "time_opt", "eccentricity_opt", "total_towers_opt",
                                     "time_e_sc_mept",
                                     "eccentricity_e_sc_mept", "total_towers_e_sc_mept", "time_e_t_mept",
                                     "eccentricity_e_t_mept",
                                     "total_towers_e_t_mept"])
                        # creazione path di salvataggio
                        destination = get_exp_name(2, min_rad, max_rad, tower, area, 0, 0, n, size, dict_sc)
                        for i in range(1, iterations+1):
                            tower, lattice, star = problem.preprocessing_scenario(2, tower, 0, 0)

                            print(f"Iteration {i}/{iterations}")
                            config = {
                                "area_side": area,
                                "towers": tower,
                                "radius_min": min_rad,
                                "radius_max": max_rad,
                                "trajectories": n,
                                "min_dist_trajectory": size,
                                "scenario": 2,
                                "lattice_neighbors": 0,
                                "star_edges": 0,
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
                            print(
                                f"solution retrieving time OPT: {round(output_opt['elapsed_time'], 4)} s, MPT_SC: {round(output_v1['elapsed_time'], 4)}, MEPT_T: {round(output_v2['elapsed_time'], 4)}")
                            result_row = [i, output_opt["elapsed_time"], output_opt["eccentricity"],
                                          output_opt["total_towers"],
                                          output_v1["elapsed_time"], output_v1["eccentricity"],
                                          output_v1["total_towers"],
                                          output_v2["elapsed_time"], output_v2["eccentricity"],
                                          output_v2["total_towers"],
                                          ]
                            output.loc[len(output)] = result_row
                        output.to_csv(destination)
    print(f"RGG with variable radius completed.")
    return


def run_regular_manhattan(areas, towers, n_traj, traj_sizes, iterations, dict_sc, debug):
    print(f"Regular Manhattan started...")
    for area in areas:
        for tower in towers:
            for n in n_traj:
                for size in traj_sizes:
                    # tower, lattice, star = problem.preprocessing_scenario(3, tower, 0, 0)
                    print("fare il controllo delle torri prima di far girare")
                    output = pd.DataFrame(
                        columns=["iteration_seed", "time_opt", "eccentricity_opt", "total_towers_opt", "time_e_sc_mept",
                                 "eccentricity_e_sc_mept", "total_towers_e_sc_mept", "time_e_t_mept",
                                 "eccentricity_e_t_mept",
                                 "total_towers_e_t_mept"])
                    print(f"exp area {area}, towers {tower}, n traj {n}, min traj size {size}.")
                    # creazione path di salvataggio
                    destination = get_exp_name(3, 0, 0, tower, area, 0, 0, n, size, dict_sc)
                    for i in range(1, iterations+1):
                        print(f"Iteration {i}/{iterations}")
                        config = {
                            "area_side": area,
                            "towers": tower,
                            "radius_min": 0,
                            "radius_max": 0,
                            "trajectories": n,
                            "min_dist_trajectory": size,
                            "scenario": 3,
                            "lattice_neighbors": 0,
                            "star_edges": 0,
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
                        print(
                            f"solution retrieving time OPT: {round(output_opt['elapsed_time'], 4)} s, MPT_SC: {round(output_v1['elapsed_time'], 4)}, MEPT_T: {round(output_v2['elapsed_time'], 4)}")
                        result_row = [i, output_opt["elapsed_time"], output_opt["eccentricity"],
                                      output_opt["total_towers"],
                                      output_v1["elapsed_time"], output_v1["eccentricity"],
                                      output_v1["total_towers"],
                                      output_v2["elapsed_time"], output_v2["eccentricity"],
                                      output_v2["total_towers"],
                                      ]
                        output.loc[len(output)] = result_row
                    output.to_csv(destination)
    print(f"Regular Manhattan completed.")
    return


def run_regular_diagonal(areas, towers, n_traj, traj_sizes, iterations, dict_sc, debug):
    print(f"Regular Diagonal started...")
    for area in areas:
        for tower in towers:
            for n in n_traj:
                for size in traj_sizes:
                    # tower, lattice, star = problem.preprocessing_scenario(4, tower, 0, 0)
                    print("fare il controllo delle torri prima di far girare")
                    output = pd.DataFrame(
                        columns=["iteration_seed", "time_opt", "eccentricity_opt", "total_towers_opt", "time_e_sc_mept",
                                 "eccentricity_e_sc_mept", "total_towers_e_sc_mept", "time_e_t_mept",
                                 "eccentricity_e_t_mept",
                                 "total_towers_e_t_mept"])
                    print(f"exp area {area}, towers {tower}, n traj {n}, min traj size {size}.")
                    # creazione path di salvataggio
                    destination = get_exp_name(4, 0, 0, tower, area, 0, 0, n, size, dict_sc)
                    for i in range(1, iterations+1):
                        print(f"Iteration {i}/{iterations}")
                        config = {
                            "area_side": area,
                            "towers": tower,
                            "radius_min": 0,
                            "radius_max": 0,
                            "trajectories": n,
                            "min_dist_trajectory": size,
                            "scenario": 4,
                            "lattice_neighbors": 0,
                            "star_edges": 0,
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
                        print(
                            f"solution retrieving time OPT: {round(output_opt['elapsed_time'], 4)} s, MPT_SC: {round(output_v1['elapsed_time'], 4)}, MEPT_T: {round(output_v2['elapsed_time'], 4)}")
                        result_row = [i, output_opt["elapsed_time"], output_opt["eccentricity"],
                                      output_opt["total_towers"],
                                      output_v1["elapsed_time"], output_v1["eccentricity"],
                                      output_v1["total_towers"],
                                      output_v2["elapsed_time"], output_v2["eccentricity"],
                                      output_v2["total_towers"],
                                      ]
                        output.loc[len(output)] = result_row
                    output.to_csv(destination)
    print(f"Regular Diagonal completed.")
    return


def run_lattice(areas, towers, lattices, n_traj, traj_sizes, iterations, dict_sc, debug):
    print(f"Lattice started...")
    for area in areas:
        for tower in towers:
            for n in n_traj:
                for size in traj_sizes:
                    for lattice in lattices:
                        print("fare il controllo delle torri prima di far girare")
                        if lattice > tower / 2:
                            continue
                        # tower, lattice, star = problem.preprocessing_scenario(6, tower, lattice, 0)
                        output = pd.DataFrame(
                            columns=["iteration_seed", "time_opt", "eccentricity_opt", "total_towers_opt",
                                     "time_e_sc_mept",
                                     "eccentricity_e_sc_mept", "total_towers_e_sc_mept", "time_e_t_mept",
                                     "eccentricity_e_t_mept",
                                     "total_towers_e_t_mept"])
                        print(f"exp area {area}, towers {tower}, neighbors {lattice}, n traj {n}, min traj size {size}.")
                        # creazione path di salvataggio
                        destination = get_exp_name(6, 0, 0, tower, area, lattice, 0, n, size, dict_sc)
                        print(f"dir: {destination}")
                        for i in range(1, iterations+1):
                            print(f"Iteration {i}/{iterations}")
                            config = {
                                "area_side": area,
                                "towers": tower,
                                "radius_min": 0,
                                "radius_max": 0,
                                "trajectories": n,
                                "min_dist_trajectory": size,
                                "scenario": 6,
                                "lattice_neighbors": lattice,
                                "star_edges": 0,
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
                            print(
                                f"solution retrieving time OPT: {round(output_opt['elapsed_time'], 4)} s, MPT_SC: {round(output_v1['elapsed_time'], 4)}, MEPT_T: {round(output_v2['elapsed_time'], 4)}")
                            result_row = [i, output_opt["elapsed_time"], output_opt["eccentricity"],
                                          output_opt["total_towers"],
                                          output_v1["elapsed_time"], output_v1["eccentricity"],
                                          output_v1["total_towers"],
                                          output_v2["elapsed_time"], output_v2["eccentricity"],
                                          output_v2["total_towers"],
                                          ]
                            output.loc[len(output)] = result_row
                        output.to_csv(destination)
    print(f"Lattice completed.")
    return


def run_star(areas, towers, stars, n_traj, traj_sizes, iterations, dict_sc, debug):
    print(f"Lattice started...")
    for area in areas:
        for tower in towers:
            for n in n_traj:
                for size in traj_sizes:
                    for s in stars:
                        # tower, lattice, star = problem.preprocessing_scenario(7, tower, 0, s)
                        print("fare il controllo delle torri prima di far girare")
                        output = pd.DataFrame(
                            columns=["iteration_seed", "time_opt", "eccentricity_opt", "total_towers_opt",
                                     "time_e_sc_mept",
                                     "eccentricity_e_sc_mept", "total_towers_e_sc_mept", "time_e_t_mept",
                                     "eccentricity_e_t_mept",
                                     "total_towers_e_t_mept"])
                        print(f"exp area {area}, towers {tower}, stars {s}, n traj {n}, min traj size {size}.")
                        # creazione path di salvataggio
                        destination = get_exp_name(7, 0, 0, tower, area, 0, s, n, size, dict_sc)
                        for i in range(1, iterations+1):
                            print(f"Iteration {i}/{iterations}")
                            config = {
                                "area_side": area,
                                "towers": tower,
                                "radius_min": 0,
                                "radius_max": 0,
                                "trajectories": n,
                                "min_dist_trajectory": size,
                                "scenario": 7,
                                "lattice_neighbors": 0,
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
                            print(
                                f"solution retrieving time OPT: {round(output_opt['elapsed_time'], 4)} s, MPT_SC: {round(output_v1['elapsed_time'], 4)}, MEPT_T: {round(output_v2['elapsed_time'], 4)}")
                            result_row = [i, output_opt["elapsed_time"], output_opt["eccentricity"],
                                          output_opt["total_towers"],
                                          output_v1["elapsed_time"], output_v1["eccentricity"],
                                          output_v1["total_towers"],
                                          output_v2["elapsed_time"], output_v2["eccentricity"],
                                          output_v2["total_towers"],
                                          ]
                            output.loc[len(output)] = result_row
                        output.to_csv(destination)
    print(f"Lattice completed.")
    return


def visualize_exp_paper():
    exp_folder = "exp"

    scenarios_folder = os.listdir(exp_folder)

        
    for dir in scenarios_folder:
        current_path = os.path.join(exp_folder, dir)
        if not os.path.exists(os.path.join(current_path, 'img')):
            os.makedirs(os.path.join(current_path, 'img'))
        files = os.listdir(current_path)
        files.remove('img')
        for file in files:
            current_file = os.path.join(current_path, file)
            print(current_file)
            plot_experiment_results(current_file)
    return

#the function below is a temporary function which was implemented to solve a temporary issue
def fix_exp_results():
    print("ATTENTION: this method should be invoked only if the CSV of experiments are broken.\n")
    scenarios = [1, 2, 3, 4, 7]
    # scenarios = [1]
    for scenario in scenarios:
        print("============================")
        print(f"folder: {scenario}")
        folder = os.path.join("exp", dir_dict[scenario])
        dir_wrong = os.path.join(folder, "wrong")
        wrong_files = os.listdir(dir_wrong)
        names_rows = []
        for file in wrong_files:
            file_path = os.path.join(dir_wrong, file)
            csv = pd.read_csv(file_path)
            names_rows.append([file_path, csv.shape[0]])
            # print(f"name: {file}-> #rows: {csv.shape[0]}")
        print(f"unsorted: {names_rows}")
        names_rows.sort(key=lambda x: x[1])
        print(f"sorted: {names_rows}")
        print("============================")

        larger_item = names_rows[-1]
        n_csv = int(larger_item[1] / 33)
        larger_name = larger_item[0]
        larger_csv = pd.read_csv(larger_name)
        print(f"larger: {larger_item}, #CSV: {n_csv}/{len(names_rows)}")
        for idx in range(n_csv):
            original_wrong_name = names_rows[idx][0]
            parts = original_wrong_name.split(os.sep)
            parts.remove('wrong')
            correct_file = os.path.join(*parts)
            print(f"right folder: {correct_file}")
            start = idx * 33
            end = start + 33
            correct_csv = larger_csv.iloc[start:end, :]
            # print(correct_csv.head())
            correct_csv = correct_csv.drop('Unnamed: 0', axis=1)
            correct_csv.to_csv(correct_file, index=False)
    return

def visualize_exp_aggregate():
    trajectories = [1, 10, 20, 50, 100]

    #=================== Aggregate RGG fixed =========================================
    folder_rgg_fixed = os.path.join("exp", "rgg_fixed", "aggregated")
    folder_rgg_fixed_towers = os.path.join(folder_rgg_fixed, "towers")
    folder_rgg_fixed_radius = os.path.join(folder_rgg_fixed, "radius")
    if not os.path.exists(folder_rgg_fixed):
        os.makedirs(folder_rgg_fixed)
        os.makedirs(folder_rgg_fixed_towers)
        os.makedirs(folder_rgg_fixed_radius)

    towers = [50, 100, 200, 350]
    radius = [100, 150, 200, 250, 300]

    #Aggregation with respect to towers number
    # for rad in radius:
    #     for tower in towers:
    #         lst_times_opt = []
    #         lst_times_sc = []
    #         lst_times_t = []
    #
    #         lst_towers_opt = []
    #         lst_towers_sc = []
    #         lst_towers_t = []
    #
    #         lst_time_confidences_opt = []
    #         lst_time_confidences_sc = []
    #         lst_time_confidences_t = []
    #
    #         lst_tower_confidences_opt = []
    #         lst_tower_confidences_sc = []
    #         lst_tower_confidences_t = []
    #         res_name = os.path.join(folder_rgg_fixed_towers, f"aggTowers_rad{rad}_nt{tower}.png")
    #         for trj in trajectories:
    #             if (tower < 100 and rad < 200) or (tower < 150 and rad < 150):
    #                 lst_times_opt.append(0)
    #                 lst_times_sc.append(0)
    #                 lst_times_t.append(0)
    #
    #                 lst_time_confidences_opt.append((0, 0))
    #                 lst_time_confidences_sc.append((0, 0))
    #                 lst_time_confidences_t.append((0, 0))
    #
    #                 lst_towers_opt.append(0)
    #                 lst_towers_sc.append(0)
    #                 lst_towers_t.append(0)
    #
    #                 lst_tower_confidences_opt.append((0, 0))
    #                 lst_tower_confidences_sc.append((0, 0))
    #                 lst_tower_confidences_t.append((0, 0))
    #                 continue
    #
    #             csv_name = f"result_a1000_t{tower}_r{rad}_nt{trj}_ts666.csv"
    #             csv = pd.read_csv(os.path.join("exp", "rgg_fixed", csv_name))
    #
    #             lst_times_opt.append(csv["time_opt"].mean())
    #             lst_times_sc.append(csv["time_e_sc_mept"].mean())
    #             lst_times_t.append(csv["time_e_t_mept"].mean())
    #
    #             lst_time_confidences_opt.append(st.t.interval(alpha=0.95, df=len(csv["time_opt"]) - 1, loc=np.mean(csv["time_opt"]), scale=st.sem(csv["time_opt"])))
    #             lst_time_confidences_sc.append(st.t.interval(alpha=0.95, df=len(csv["time_e_sc_mept"]) - 1, loc=np.mean(csv["time_e_sc_mept"]), scale=st.sem(csv["time_e_sc_mept"])))
    #             lst_time_confidences_t.append(st.t.interval(alpha=0.95, df=len(csv["time_e_t_mept"]) - 1, loc=np.mean(csv["time_e_t_mept"]), scale=st.sem(csv["time_e_t_mept"])))
    #
    #             lst_towers_opt.append(csv["total_towers_opt"].mean())
    #             lst_towers_sc.append(csv["total_towers_e_sc_mept"].mean())
    #             lst_towers_t.append(csv["total_towers_e_t_mept"].mean())
    #
    #             lst_tower_confidences_opt.append(st.t.interval(alpha=0.95, df=len(csv["time_opt"]) - 1, loc=np.mean(csv["time_opt"]),
    #                               scale=st.sem(csv["time_opt"])))
    #             lst_tower_confidences_sc.append(st.t.interval(alpha=0.95, df=len(csv["time_e_sc_mept"]) - 1, loc=np.mean(csv["time_e_sc_mept"]),
    #                               scale=st.sem(csv["time_e_sc_mept"])))
    #             lst_tower_confidences_t.append(st.t.interval(alpha=0.95, df=len(csv["time_e_t_mept"]) - 1, loc=np.mean(csv["time_e_t_mept"]),
    #                               scale=st.sem(csv["time_e_t_mept"])))
    #
    #         plot_aggregate(res_name, trajectories, [lst_towers_opt, lst_towers_sc, lst_towers_t], [lst_times_opt, lst_times_sc, lst_times_t], [lst_tower_confidences_opt, lst_tower_confidences_sc, lst_tower_confidences_t], [lst_time_confidences_opt, lst_time_confidences_sc, lst_time_confidences_t])
    #
    #     # Aggregation with respect to towers number
    #
    # for tower in towers:
    #     for rad in radius:
    #         lst_times_opt = []
    #         lst_times_sc = []
    #         lst_times_t = []
    #
    #         lst_towers_opt = []
    #         lst_towers_sc = []
    #         lst_towers_t = []
    #
    #         lst_time_confidences_opt = []
    #         lst_time_confidences_sc = []
    #         lst_time_confidences_t = []
    #
    #         lst_tower_confidences_opt = []
    #         lst_tower_confidences_sc = []
    #         lst_tower_confidences_t = []
    #         res_name = os.path.join(folder_rgg_fixed_radius, f"aggRad_rad{rad}_nt{tower}.png")
    #         for trj in trajectories:
    #             if (tower < 100 and rad < 200) or (tower < 150 and rad < 150):
    #                 lst_times_opt.append(0)
    #                 lst_times_sc.append(0)
    #                 lst_times_t.append(0)
    #
    #                 lst_time_confidences_opt.append((0, 0))
    #                 lst_time_confidences_sc.append((0, 0))
    #                 lst_time_confidences_t.append((0, 0))
    #
    #                 lst_towers_opt.append(0)
    #                 lst_towers_sc.append(0)
    #                 lst_towers_t.append(0)
    #
    #                 lst_tower_confidences_opt.append((0, 0))
    #                 lst_tower_confidences_sc.append((0, 0))
    #                 lst_tower_confidences_t.append((0, 0))
    #                 continue
    #
    #             csv_name = f"result_a1000_t{tower}_r{rad}_nt{trj}_ts666.csv"
    #             csv = pd.read_csv(os.path.join("exp", "rgg_fixed", csv_name))
    #
    #             lst_times_opt.append(csv["time_opt"].mean())
    #             lst_times_sc.append(csv["time_e_sc_mept"].mean())
    #             lst_times_t.append(csv["time_e_t_mept"].mean())
    #
    #             lst_time_confidences_opt.append(
    #                 st.t.interval(alpha=0.95, df=len(csv["time_opt"]) - 1, loc=np.mean(csv["time_opt"]),
    #                               scale=st.sem(csv["time_opt"])))
    #             lst_time_confidences_sc.append(
    #                 st.t.interval(alpha=0.95, df=len(csv["time_e_sc_mept"]) - 1, loc=np.mean(csv["time_e_sc_mept"]),
    #                               scale=st.sem(csv["time_e_sc_mept"])))
    #             lst_time_confidences_t.append(
    #                 st.t.interval(alpha=0.95, df=len(csv["time_e_t_mept"]) - 1, loc=np.mean(csv["time_e_t_mept"]),
    #                               scale=st.sem(csv["time_e_t_mept"])))
    #
    #             lst_towers_opt.append(csv["total_towers_opt"].mean())
    #             lst_towers_sc.append(csv["total_towers_e_sc_mept"].mean())
    #             lst_towers_t.append(csv["total_towers_e_t_mept"].mean())
    #
    #             lst_tower_confidences_opt.append(
    #                 st.t.interval(alpha=0.95, df=len(csv["time_opt"]) - 1, loc=np.mean(csv["time_opt"]),
    #                               scale=st.sem(csv["time_opt"])))
    #             lst_tower_confidences_sc.append(
    #                 st.t.interval(alpha=0.95, df=len(csv["time_e_sc_mept"]) - 1, loc=np.mean(csv["time_e_sc_mept"]),
    #                               scale=st.sem(csv["time_e_sc_mept"])))
    #             lst_tower_confidences_t.append(
    #                 st.t.interval(alpha=0.95, df=len(csv["time_e_t_mept"]) - 1, loc=np.mean(csv["time_e_t_mept"]),
    #                               scale=st.sem(csv["time_e_t_mept"])))
    #
    #         plot_aggregate(res_name, trajectories, [lst_towers_opt, lst_towers_sc, lst_towers_t],
    #                        [lst_times_opt, lst_times_sc, lst_times_t],
    #                        [lst_tower_confidences_opt, lst_tower_confidences_sc, lst_tower_confidences_t],
    #                        [lst_time_confidences_opt, lst_time_confidences_sc, lst_time_confidences_t])

    # ====================================================================================
    # =================== Aggregate RGG variable =========================================
    # folder_rgg_variable = os.path.join("exp", "rgg_variable", "aggregated")
    # folder_rgg_varible_towers = os.path.join(folder_rgg_variable, "towers")
    # folder_rgg_variable_radius = os.path.join(folder_rgg_variable, "radius")
    # if not os.path.exists(folder_rgg_variable):
    #     os.makedirs(folder_rgg_variable)
    #     os.makedirs(folder_rgg_varible_towers)
    #     os.makedirs(folder_rgg_variable_radius)

    # towers = [50, 100, 200, 350]
    # radius = [(100, 300), (150, 300), (200, 300), (250, 300)]
    #
    # #Aggregation with respect to towers number
    # for rad1, rad2 in radius:
    #     for tower in towers:
    #         lst_times_opt = []
    #         lst_times_sc = []
    #         lst_times_t = []
    #
    #         lst_towers_opt = []
    #         lst_towers_sc = []
    #         lst_towers_t = []
    #
    #         lst_time_confidences_opt = []
    #         lst_time_confidences_sc = []
    #         lst_time_confidences_t = []
    #
    #         lst_tower_confidences_opt = []
    #         lst_tower_confidences_sc = []
    #         lst_tower_confidences_t = []
    #         res_name = os.path.join(folder_rgg_varible_towers, f"aggTowers_rad{(rad1, rad2)}_nt{tower}.png")
    #         for trj in trajectories:
    #             if tower < 100 and rad1 < 200:
    #                 lst_times_opt.append(0)
    #                 lst_times_sc.append(0)
    #                 lst_times_t.append(0)
    #
    #                 lst_time_confidences_opt.append((0, 0))
    #                 lst_time_confidences_sc.append((0, 0))
    #                 lst_time_confidences_t.append((0, 0))
    #
    #                 lst_towers_opt.append(0)
    #                 lst_towers_sc.append(0)
    #                 lst_towers_t.append(0)
    #
    #                 lst_tower_confidences_opt.append((0, 0))
    #                 lst_tower_confidences_sc.append((0, 0))
    #                 lst_tower_confidences_t.append((0, 0))
    #                 continue
    #
    #             csv_name = f"result_a1000_t{tower}_rmin{rad1}_rmax{rad2}_nt{trj}_ts666.csv"
    #             csv = pd.read_csv(os.path.join("exp", "rgg_variable", csv_name))
    #
    #             lst_times_opt.append(csv["time_opt"].mean())
    #             lst_times_sc.append(csv["time_e_sc_mept"].mean())
    #             lst_times_t.append(csv["time_e_t_mept"].mean())
    #
    #             lst_time_confidences_opt.append(st.t.interval(alpha=0.95, df=len(csv["time_opt"]) - 1, loc=np.mean(csv["time_opt"]), scale=st.sem(csv["time_opt"])))
    #             lst_time_confidences_sc.append(st.t.interval(alpha=0.95, df=len(csv["time_e_sc_mept"]) - 1, loc=np.mean(csv["time_e_sc_mept"]), scale=st.sem(csv["time_e_sc_mept"])))
    #             lst_time_confidences_t.append(st.t.interval(alpha=0.95, df=len(csv["time_e_t_mept"]) - 1, loc=np.mean(csv["time_e_t_mept"]), scale=st.sem(csv["time_e_t_mept"])))
    #
    #             lst_towers_opt.append(csv["total_towers_opt"].mean())
    #             lst_towers_sc.append(csv["total_towers_e_sc_mept"].mean())
    #             lst_towers_t.append(csv["total_towers_e_t_mept"].mean())
    #
    #             lst_tower_confidences_opt.append(st.t.interval(alpha=0.95, df=len(csv["time_opt"]) - 1, loc=np.mean(csv["time_opt"]),
    #                               scale=st.sem(csv["time_opt"])))
    #             lst_tower_confidences_sc.append(st.t.interval(alpha=0.95, df=len(csv["time_e_sc_mept"]) - 1, loc=np.mean(csv["time_e_sc_mept"]),
    #                               scale=st.sem(csv["time_e_sc_mept"])))
    #             lst_tower_confidences_t.append(st.t.interval(alpha=0.95, df=len(csv["time_e_t_mept"]) - 1, loc=np.mean(csv["time_e_t_mept"]),
    #                               scale=st.sem(csv["time_e_t_mept"])))
    #
    #         plot_aggregate(res_name, trajectories, [lst_towers_opt, lst_towers_sc, lst_towers_t], [lst_times_opt, lst_times_sc, lst_times_t], [lst_tower_confidences_opt, lst_tower_confidences_sc, lst_tower_confidences_t], [lst_time_confidences_opt, lst_time_confidences_sc, lst_time_confidences_t])
    #
    # # Aggregation with respect to towers number
    # for tower in towers:
    #     for rad1, rad2 in radius:
    #         lst_times_opt = []
    #         lst_times_sc = []
    #         lst_times_t = []
    #
    #         lst_towers_opt = []
    #         lst_towers_sc = []
    #         lst_towers_t = []
    #
    #         lst_time_confidences_opt = []
    #         lst_time_confidences_sc = []
    #         lst_time_confidences_t = []
    #
    #         lst_tower_confidences_opt = []
    #         lst_tower_confidences_sc = []
    #         lst_tower_confidences_t = []
    #         res_name = os.path.join(folder_rgg_variable_radius, f"aggRad_rad{(rad1, rad2)}_nt{tower}.png")
    #         for trj in trajectories:
    #             if tower < 100 and rad1 < 200:
    #                 lst_times_opt.append(0)
    #                 lst_times_sc.append(0)
    #                 lst_times_t.append(0)
    #
    #                 lst_time_confidences_opt.append((0, 0))
    #                 lst_time_confidences_sc.append((0, 0))
    #                 lst_time_confidences_t.append((0, 0))
    #
    #                 lst_towers_opt.append(0)
    #                 lst_towers_sc.append(0)
    #                 lst_towers_t.append(0)
    #
    #                 lst_tower_confidences_opt.append((0, 0))
    #                 lst_tower_confidences_sc.append((0, 0))
    #                 lst_tower_confidences_t.append((0, 0))
    #                 continue
    #
    #             csv_name = f"result_a1000_t{tower}_rmin{rad1}_rmax{rad2}_nt{trj}_ts666.csv"
    #             csv = pd.read_csv(os.path.join("exp", "rgg_variable", csv_name))
    #
    #             lst_times_opt.append(csv["time_opt"].mean())
    #             lst_times_sc.append(csv["time_e_sc_mept"].mean())
    #             lst_times_t.append(csv["time_e_t_mept"].mean())
    #
    #             lst_time_confidences_opt.append(
    #                 st.t.interval(alpha=0.95, df=len(csv["time_opt"]) - 1, loc=np.mean(csv["time_opt"]),
    #                               scale=st.sem(csv["time_opt"])))
    #             lst_time_confidences_sc.append(
    #                 st.t.interval(alpha=0.95, df=len(csv["time_e_sc_mept"]) - 1, loc=np.mean(csv["time_e_sc_mept"]),
    #                               scale=st.sem(csv["time_e_sc_mept"])))
    #             lst_time_confidences_t.append(
    #                 st.t.interval(alpha=0.95, df=len(csv["time_e_t_mept"]) - 1, loc=np.mean(csv["time_e_t_mept"]),
    #                               scale=st.sem(csv["time_e_t_mept"])))
    #
    #             lst_towers_opt.append(csv["total_towers_opt"].mean())
    #             lst_towers_sc.append(csv["total_towers_e_sc_mept"].mean())
    #             lst_towers_t.append(csv["total_towers_e_t_mept"].mean())
    #
    #             lst_tower_confidences_opt.append(
    #                 st.t.interval(alpha=0.95, df=len(csv["time_opt"]) - 1, loc=np.mean(csv["time_opt"]),
    #                               scale=st.sem(csv["time_opt"])))
    #             lst_tower_confidences_sc.append(
    #                 st.t.interval(alpha=0.95, df=len(csv["time_e_sc_mept"]) - 1, loc=np.mean(csv["time_e_sc_mept"]),
    #                               scale=st.sem(csv["time_e_sc_mept"])))
    #             lst_tower_confidences_t.append(
    #                 st.t.interval(alpha=0.95, df=len(csv["time_e_t_mept"]) - 1, loc=np.mean(csv["time_e_t_mept"]),
    #                               scale=st.sem(csv["time_e_t_mept"])))
    #
    #         plot_aggregate(res_name, trajectories, [lst_towers_opt, lst_towers_sc, lst_towers_t],
    #                        [lst_times_opt, lst_times_sc, lst_times_t],
    #                        [lst_tower_confidences_opt, lst_tower_confidences_sc, lst_tower_confidences_t],
    #                        [lst_time_confidences_opt, lst_time_confidences_sc, lst_time_confidences_t])
    # ====================================================================================


    #====================== Manhattan Diagonal Together ==================================
    # towers = [4 ** 2, 6 ** 2, 7 ** 2, 8 ** 2, 10 ** 2, 12 ** 2, 14 ** 2, 15 ** 2, 16 ** 2]
    # prefix_dia = os.path.join("exp", "diagonal")
    # prefix_mana = os.path.join("exp", "manhattan")
    # for tower in towers:
    #     df_agg = pd.DataFrame(columns=["trajectories", "diagonal_time_opt", "diagonal_towers_opt", "diagonal_time_sc",
    #                                    "diagonal_towers_sc", "diagonal_time_t", "diagonal_towers_t", "manhattan_time_opt",
    #                                    "manhattan_towers_opt", "manhattan_time_sc", "manhattan_towers_sc", "manhattan_time_t",
    #                                    "manhattan_towers_t", "diagonal_time_opt_conf", "diagonal_towers_opt_conf", "diagonal_time_sc_conf",
    #                                    "diagonal_towers_sc_conf", "diagonal_time_t_conf", "diagonal_towers_t_conf", "manhattan_time_opt_conf",
    #                                    "manhattan_towers_opt_conf", "manhattan_time_sc_conf", "manhattan_towers_sc_conf", "manhattan_time_t_conf",
    #                                    "manhattan_towers_t_conf"])
    #     for trj in trajectories:
    #         file_name = f"result_a1000_t{tower}_nt{trj}_ts666.csv"
    #
    #         csv_dia = pd.read_csv(os.path.join(prefix_dia, file_name))
    #         csv_mana = pd.read_csv(os.path.join(prefix_mana, file_name))
    #
    #         row = [
    #             trj, csv_dia["time_opt"].mean(), csv_dia["total_towers_opt"].mean(), csv_dia["time_e_sc_mept"].mean(),
    #             csv_dia["total_towers_e_sc_mept"].mean(), csv_dia["time_e_t_mept"].mean(), csv_dia["total_towers_e_t_mept"].mean(),
    #             csv_mana["time_opt"].mean(), csv_mana["total_towers_opt"].mean(), csv_mana["time_e_sc_mept"].mean(),
    #             csv_mana["total_towers_e_sc_mept"].mean(), csv_mana["time_e_t_mept"].mean(), csv_mana["total_towers_e_t_mept"].mean(),
    #             get_confidence(csv_dia["time_opt"].tolist()), get_confidence(csv_dia["total_towers_opt"].tolist()), get_confidence(csv_dia["time_e_sc_mept"].tolist()),
    #             get_confidence(csv_dia["total_towers_e_sc_mept"].tolist()), get_confidence(csv_dia["time_e_t_mept"].tolist()),
    #             get_confidence(csv_dia["total_towers_e_t_mept"].tolist()),
    #             get_confidence(csv_mana["time_opt"].tolist()), get_confidence(csv_mana["total_towers_opt"].tolist()), get_confidence(csv_mana["time_e_sc_mept"].tolist()),
    #             get_confidence(csv_mana["total_towers_e_sc_mept"].tolist()), get_confidence(csv_mana["time_e_t_mept"].tolist()),
    #             get_confidence(csv_mana["total_towers_e_t_mept"].tolist())]
    #         df_agg.loc[len(df_agg.index)] = row
    #     if not os.path.exists(os.path.join(prefix_dia, "aggregated")):
    #         os.makedirs(os.path.join(prefix_dia, "aggregated"))
    #     dst_name = os.path.join(prefix_dia, "aggregated", f"aggTower_dia_mana_t{tower}.png")
    #     plot_algorithm_diagonal_manhattan(df_agg, dst_name)
    
    # ====================================================================================

    #================================ Lattice ============================================
    # towers = [5, 10, 15, 20]
    # lattice_neighbors = [2, 4, 6]
    #
    # folder_lattice = os.path.join("exp", "lattice", "aggregated")
    # if not os.path.exists(folder_lattice):
    #     os.makedirs(folder_lattice)
    #
    # for tower in towers:
    #     for lattice in lattice_neighbors:
    #         res = pd.DataFrame(columns=[
    #             "trajectories", "opt_time", "opt_towers", "e_sc_mept_time", "e_sc_mept_towers", "e_t_mept_time",
    #             "e_t_mept_towers", "opt_time_conf", "opt_towers_conf", "e_sc_mept_time_conf", "e_sc_mept_towers_conf",
    #             "e_t_mept_time_conf", "e_t_mept_towers_conf"
    #         ])
    #         for trj in trajectories:
    #             if lattice > tower / 2:
    #                 res.loc[len(res.index)] = [trj, 0, 0, 0, 0, 0, 0, (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    #                 continue
    #             csv_name = f"result_a1000_t{tower}_neig{lattice}_nt{trj}_ts666.csv"
    #             csv = pd.read_csv(os.path.join("exp", "lattice", csv_name))
    #             res.loc[len(res.index)] = [
    #                 trj, csv["time_opt"].mean(), csv["total_towers_opt"].mean(), csv["time_e_sc_mept"].mean(), csv["total_towers_e_sc_mept"].mean(),
    #                 csv["time_e_t_mept"].mean(), csv["total_towers_e_t_mept"].mean(), get_confidence(csv["time_opt"].tolist()), get_confidence(csv["total_towers_opt"].tolist()),
    #                 get_confidence(csv["time_e_sc_mept"].tolist()), get_confidence(csv["total_towers_e_sc_mept"].tolist()), get_confidence(csv["time_e_t_mept"].tolist()), get_confidence(csv["total_towers_e_t_mept"].tolist())
    #                    ]
    #         res_name = f"aggNeigh_ng{lattice}_t{tower}.png"
    #         plot_aggregate(os.path.join(folder_lattice, res_name), trajectories, [res["opt_towers"].tolist(), res["e_sc_mept_towers"].tolist(), res["e_t_mept_towers"].tolist()], [res["opt_time"].tolist(), res["e_sc_mept_time"].tolist(), res["e_t_mept_time"].tolist()],
    #                        [res["opt_towers_conf"], res["e_sc_mept_towers_conf"], res["e_t_mept_towers_conf"]],
    #                        [res["opt_time_conf"].tolist(), res["e_sc_mept_time_conf"].tolist(), res["e_t_mept_time_conf"].tolist()])
    # for lattice in lattice_neighbors:
    #     for tower in towers:
    #         res = pd.DataFrame(columns=[
    #             "trajectories", "opt_time", "opt_towers", "e_sc_mept_time", "e_sc_mept_towers", "e_t_mept_time",
    #             "e_t_mept_towers", "opt_time_conf", "opt_towers_conf", "e_sc_mept_time_conf", "e_sc_mept_towers_conf",
    #             "e_t_mept_time_conf", "e_t_mept_towers_conf"
    #         ])
    #         for trj in trajectories:
    #             if lattice > tower / 2:
    #                 res.loc[len(res.index)] = [trj, 0, 0, 0, 0, 0, 0, (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    #                 continue
    #             csv_name = f"result_a1000_t{tower}_neig{lattice}_nt{trj}_ts666.csv"
    #             csv = pd.read_csv(os.path.join("exp", "lattice", csv_name))
    #             res.loc[len(res.index)] = [
    #                 trj, csv["time_opt"].mean(), csv["total_towers_opt"].mean(), csv["time_e_sc_mept"].mean(), csv["total_towers_e_sc_mept"].mean(),
    #                 csv["time_e_t_mept"].mean(), csv["total_towers_e_t_mept"].mean(), get_confidence(csv["time_opt"].tolist()), get_confidence(csv["total_towers_opt"].tolist()),
    #                 get_confidence(csv["time_e_sc_mept"].tolist()), get_confidence(csv["total_towers_e_sc_mept"].tolist()), get_confidence(csv["time_e_t_mept"].tolist()), get_confidence(csv["total_towers_e_t_mept"].tolist())
    #                    ]
    #         res_name = f"aggTower_ng{lattice}_t{tower}.png"
    #         plot_aggregate(os.path.join(folder_lattice, res_name), trajectories, [res["opt_towers"].tolist(), res["e_sc_mept_towers"].tolist(), res["e_t_mept_towers"].tolist()], [res["opt_time"].tolist(), res["e_sc_mept_time"].tolist(), res["e_t_mept_time"].tolist()],
    #                        [res["opt_towers_conf"], res["e_sc_mept_towers_conf"], res["e_t_mept_towers_conf"]],
    #                        [res["opt_time_conf"].tolist(), res["e_sc_mept_time_conf"].tolist(), res["e_t_mept_time_conf"].tolist()])
    # ====================================================================================

    # ===================================== Star =========================================
    star_edges = [5, 7, 10, 12]
    towers = [31, 57, 111, 157]

    folder_lattice = os.path.join("exp", "star", "aggregated")
    if not os.path.exists(folder_lattice):
        os.makedirs(folder_lattice)

    for tower in towers:
        for star in star_edges:
            res = pd.DataFrame(columns=[
                "trajectories", "opt_time", "opt_towers", "e_sc_mept_time", "e_sc_mept_towers", "e_t_mept_time",
                "e_t_mept_towers", "opt_time_conf", "opt_towers_conf", "e_sc_mept_time_conf", "e_sc_mept_towers_conf",
                "e_t_mept_time_conf", "e_t_mept_towers_conf"
            ])
            for trj in trajectories:
                csv_name = f"result_a1000_t{tower}_star{star}_nt{trj}_ts666.csv"
                if not os.path.exists(os.path.join("exp", "star", csv_name)):
                    res.loc[len(res.index)] = [trj, 0, 0, 0, 0, 0, 0, (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
                    continue
                csv = pd.read_csv(os.path.join("exp", "star", csv_name))
                res.loc[len(res.index)] = [
                    trj, csv["time_opt"].mean(), csv["total_towers_opt"].mean(), csv["time_e_sc_mept"].mean(),
                    csv["total_towers_e_sc_mept"].mean(),
                    csv["time_e_t_mept"].mean(), csv["total_towers_e_t_mept"].mean(),
                    get_confidence(csv["time_opt"].tolist()), get_confidence(csv["total_towers_opt"].tolist()),
                    get_confidence(csv["time_e_sc_mept"].tolist()),
                    get_confidence(csv["total_towers_e_sc_mept"].tolist()),
                    get_confidence(csv["time_e_t_mept"].tolist()), get_confidence(csv["total_towers_e_t_mept"].tolist())
                ]
            res_name = f"aggStar_star{star}_t{tower}.png"
            plot_aggregate(os.path.join(folder_lattice, res_name), trajectories,
                           [res["opt_towers"].tolist(), res["e_sc_mept_towers"].tolist(),
                            res["e_t_mept_towers"].tolist()],
                           [res["opt_time"].tolist(), res["e_sc_mept_time"].tolist(), res["e_t_mept_time"].tolist()],
                           [res["opt_towers_conf"], res["e_sc_mept_towers_conf"], res["e_t_mept_towers_conf"]],
                           [res["opt_time_conf"].tolist(), res["e_sc_mept_time_conf"].tolist(),
                            res["e_t_mept_time_conf"].tolist()])
    for star in star_edges:
        for tower in towers:
            res = pd.DataFrame(columns=[
                "trajectories", "opt_time", "opt_towers", "e_sc_mept_time", "e_sc_mept_towers", "e_t_mept_time",
                "e_t_mept_towers", "opt_time_conf", "opt_towers_conf", "e_sc_mept_time_conf", "e_sc_mept_towers_conf",
                "e_t_mept_time_conf", "e_t_mept_towers_conf"
            ])
            for trj in trajectories:
                csv_name = f"result_a1000_t{tower}_star{star}_nt{trj}_ts666.csv"
                if not os.path.exists(os.path.join("exp", "star", csv_name)):
                    res.loc[len(res.index)] = [trj, 0, 0, 0, 0, 0, 0, (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
                    continue
                csv = pd.read_csv(os.path.join("exp", "star", csv_name))
                res.loc[len(res.index)] = [
                    trj, csv["time_opt"].mean(), csv["total_towers_opt"].mean(), csv["time_e_sc_mept"].mean(),
                    csv["total_towers_e_sc_mept"].mean(),
                    csv["time_e_t_mept"].mean(), csv["total_towers_e_t_mept"].mean(),
                    get_confidence(csv["time_opt"].tolist()), get_confidence(csv["total_towers_opt"].tolist()),
                    get_confidence(csv["time_e_sc_mept"].tolist()),
                    get_confidence(csv["total_towers_e_sc_mept"].tolist()),
                    get_confidence(csv["time_e_t_mept"].tolist()), get_confidence(csv["total_towers_e_t_mept"].tolist())
                ]
            res_name = f"aggTower_star{star}_t{tower}.png"
            plot_aggregate(os.path.join(folder_lattice, res_name), trajectories,
                           [res["opt_towers"].tolist(), res["e_sc_mept_towers"].tolist(),
                            res["e_t_mept_towers"].tolist()],
                           [res["opt_time"].tolist(), res["e_sc_mept_time"].tolist(), res["e_t_mept_time"].tolist()],
                           [res["opt_towers_conf"], res["e_sc_mept_towers_conf"], res["e_t_mept_towers_conf"]],
                           [res["opt_time_conf"].tolist(), res["e_sc_mept_time_conf"].tolist(),
                            res["e_t_mept_time_conf"].tolist()])
    # ====================================================================================


    return

