import os
import time
import pandas as pd

import problem_gen as problem
from algorithms import *
from util import get_exp_name

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
    output = pd.DataFrame(columns=["iteration_seed", "time_opt", "eccentricity_opt", "total_towers_opt", "time_e_sc_mept", "eccentricity_e_sc_mept", "total_towers_e_sc_mept", "time_e_t_mept", "eccentricity_e_t_mept", "total_towers_e_t_mept"])

    print(f"RGG with fixed radius started...")
    for area in areas:
        for tower in towers:
            for rad in rads:
                for n in n_traj:
                    for size in traj_sizes:
                        if (tower < 100 and rad < 200) or (tower < 150 and rad < 150):
                            continue

                        print(f"exp area {area}, towers {tower}, rad {rad}, n traj {n}, min traj size {size}.")
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
    output = pd.DataFrame(
        columns=["iteration_seed", "time_opt", "eccentricity_opt", "total_towers_opt", "time_e_sc_mept",
                 "eccentricity_e_sc_mept", "total_towers_e_sc_mept", "time_e_t_mept", "eccentricity_e_t_mept",
                 "total_towers_e_t_mept"])

    print(f"RGG with variable radius stated...")
    for area in areas:
        for tower in towers:
            for min_rad, max_rad in rads:
                for n in n_traj:
                    for size in traj_sizes:
                        if tower < 100 and min_rad < 200:
                            continue

                        print(f"exp area {area}, towers {tower}, rad ({min_rad},{max_rad}), n traj {n}, min traj size {size}.")
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
    output = pd.DataFrame(
        columns=["iteration_seed", "time_opt", "eccentricity_opt", "total_towers_opt", "time_e_sc_mept",
                 "eccentricity_e_sc_mept", "total_towers_e_sc_mept", "time_e_t_mept", "eccentricity_e_t_mept",
                 "total_towers_e_t_mept"])

    print(f"Regular Manhattan started...")
    for area in areas:
        for tower in towers:
            for n in n_traj:
                for size in traj_sizes:
                    tower, lattice, star = problem.preprocessing_scenario(3, tower, 0, 0)

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
    output = pd.DataFrame(
        columns=["iteration_seed", "time_opt", "eccentricity_opt", "total_towers_opt", "time_e_sc_mept",
                 "eccentricity_e_sc_mept", "total_towers_e_sc_mept", "time_e_t_mept", "eccentricity_e_t_mept",
                 "total_towers_e_t_mept"])

    print(f"Regular Diagonal started...")
    for area in areas:
        for tower in towers:
            for n in n_traj:
                for size in traj_sizes:
                    tower, lattice, star = problem.preprocessing_scenario(4, tower, 0, 0)

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
    output = pd.DataFrame(
        columns=["iteration_seed", "time_opt", "eccentricity_opt", "total_towers_opt", "time_e_sc_mept",
                 "eccentricity_e_sc_mept", "total_towers_e_sc_mept", "time_e_t_mept", "eccentricity_e_t_mept",
                 "total_towers_e_t_mept"])

    print(f"Lattice started...")
    for area in areas:
        for tower in towers:
            for n in n_traj:
                for size in traj_sizes:
                    for lattice in lattices:
                        tower, lattice, star = problem.preprocessing_scenario(6, tower, lattice, 0)

                        print(f"exp area {area}, towers {tower}, neighbors {lattice}, n traj {n}, min traj size {size}.")
                        # creazione path di salvataggio
                        destination = get_exp_name(6, 0, 0, tower, area, lattice, 0, n, size, dict_sc)
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
    output = pd.DataFrame(
        columns=["iteration_seed", "time_opt", "eccentricity_opt", "total_towers_opt", "time_e_sc_mept",
                 "eccentricity_e_sc_mept", "total_towers_e_sc_mept", "time_e_t_mept", "eccentricity_e_t_mept",
                 "total_towers_e_t_mept"])

    print(f"Lattice started...")
    for area in areas:
        for tower in towers:
            for n in n_traj:
                for size in traj_sizes:
                    for s in stars:
                        tower, lattice, star = problem.preprocessing_scenario(7, tower, 0, s)

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
        files = os.listdir(current_path)
        files.remove('img')
        for file in files:
            current_file = os.path.join(current_path, file)
            plot_experiment_results(current_file)
    return
