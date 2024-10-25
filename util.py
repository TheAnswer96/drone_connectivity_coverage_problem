import math

import networkx as nx
import pandas as pd
import os
import matplotlib.pyplot as plt
# from algorithms import *

EPSILON = 1e-5  # Small epsilon to handle floating-point precision issues


def is_square(n):
    if n < 0:
        return False

    root = math.isqrt(n)
    return root * root == n


def is_zero(value):
    return abs(value) < EPSILON


def get_distance(p0, p1):
    return math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)


def circle_segment_intersection(x0, y0, x1, y1, cx, cy, r):
    # Vector from (x0, y0) to (x1, y1)
    dx = x1 - x0
    dy = y1 - y0

    # Quadratic coefficients
    a = dx ** 2 + dy ** 2
    b = 2 * (dx * (x0 - cx) + dy * (y0 - cy))
    c = (x0 - cx) ** 2 + (y0 - cy) ** 2 - r ** 2

    # Calculate the discriminant
    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        # No intersection
        return []

    # Calculate the two points of intersection
    sqrt_discriminant = math.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)

    # Find the intersection points
    intersection_points = []

    for t in [t1, t2]:
        if 0 <= t <= 1:  # Check if the intersection is within the segment
            ix = x0 + t * dx
            iy = y0 + t * dy
            intersection_points.append((ix, iy))

    return intersection_points


def are_intervals_overlap(interval1, interval2):
    start1, end1 = interval1
    start2, end2 = interval2
    return not (end1 < start2 or end2 < start1)


def create_interval_graph(instance):
    graphs = []
    intervals = instance["intervals"]
    # Add nodes for each interval
    for interval in intervals:
        length = interval["length"]
        G = nx.Graph()
        for i, I in enumerate(interval['interval']):
            G.add_node(i, interval=I)

        # Add edges between overlapping intervals
        for i in range(len(interval['interval'])):
            for j in range(i + 1, len(interval['interval'])):
                if are_intervals_overlap(
                        [round(interval['interval'][i]['inf'], 2), round(interval['interval'][i]['sup'], 2)],
                        [round(interval['interval'][j]['inf'], 2), round(interval['interval'][j]['sup'], 2)]):
                    G.add_edge(i, j)

        # dummy nodes: starting and ending
        G.add_node(-1, interval=[0, 0])
        G.add_node(len(interval['interval']) + 1, interval=[length, length])
        for i in range(len(interval['interval'])):
            if are_intervals_overlap([0, 0], [round(interval['interval'][i]['inf'], 2),
                                              round(interval['interval'][i]['sup'], 2)]):
                G.add_edge(-1, i)
            if are_intervals_overlap([math.floor(length), math.ceil(length)], [round(interval['interval'][i]['inf'], 0),
                                                                               round(interval['interval'][i]['sup'],
                                                                                     2)]):
                G.add_edge(len(interval['interval']) + 1, i)
        graphs.append(G)
    return graphs


def is_coverage(intervals, nodes):
    # nodes is a set of tower names
    length = round(intervals["length"], 2)
    towers = []
    for interval in intervals["interval"]:
        if int(interval["tower"]) in nodes:
            towers.append([round(interval["inf"], 2), round(interval["sup"], 2), "T" + str(interval["tower"])])
    towers.sort(key=lambda x: x[0])
    # print(towers)
    last_covered = 0
    for start, end, _ in towers:
        if start > last_covered:
            return False, towers
        last_covered = max(last_covered, end)
        # print(last_covered, " ", length, " statified ", last_covered >= length)
        if last_covered >= length:
            return True, towers
    return False, towers


def get_minimum_cover(cover, length):
    sol = []
    last_covered = 0
    while last_covered < length:
        temp = [tower for tower in cover if tower[0] <= last_covered]
        temp.sort(key=lambda x: x[1])
        best_I = temp[-1]
        # print("List: ", temp, " BEST I: ", best_I)
        last_covered = max(last_covered, best_I[1])
        sol.append(best_I)
    return sol


def is_covered(dist, intervals):
    if len(intervals) == 0:
        return False

    # Step 1: Extract and sort intervals based on "inf" and "sup"
    sorted_intervals = sorted(intervals, key=lambda x: (x["inf"], x["sup"]))

    # Step 2: Merge overlapping and contiguous intervals
    min_inf = sorted_intervals[0]["inf"]
    max_sup = sorted_intervals[0]["sup"]

    # Do this when there is only one interval
    if len(intervals) == 1:
        if dist - max_sup > EPSILON:
            return False
        if min_inf > EPSILON:
            return False

    for i in range(1, len(sorted_intervals)):

        # There is a gap immediately, so exit
        if min_inf > EPSILON:
            return False

        inf = sorted_intervals[i]["inf"]
        sup = sorted_intervals[i]["sup"]

        if inf - max_sup > EPSILON:
            # There is a gap, so exit
            return False
        else:
            max_sup = max(max_sup, sup)

    if dist - max_sup > EPSILON:
        # There is a gap at the end, so exit
        return False

    return True


def create_instance_set_cover(intervals, bfs_nodes):
    i = 0
    for interval in intervals:
        length = interval["length"]
        # print(f"Trajectory T_{i} with interval [0, {length:.2f}] and {len(interval['interval'])} towers")
        endpoints = []
        for I in interval["interval"]:
            tower = I["tower"]
            if tower not in bfs_nodes:
                continue

            inf = I["inf"]
            sup = I["sup"]
            endpoints.append((inf, 'start', tower))
            endpoints.append((sup, 'end', tower))
            # print(f"  I_{tower} [{inf:.2f}, {sup:.2f}]")

        endpoints.sort()
        active_intervals = set()
        previous_point = None
        segments = []

        for point, event_type, tower in endpoints:
            if previous_point is not None and active_intervals:
                if not is_zero(point - previous_point):
                    segments.append((previous_point, point, list(active_intervals)))

            if event_type == 'start':
                active_intervals.add(tower)
            elif event_type == 'end':
                active_intervals.remove(tower)

            previous_point = point

        j = 0
        mini_intervals = []
        # print(f"The whole interval can be split into {len(segments)} mini intervals")
        for start, end, active_towers in segments:
            # print(f"  I_{i}^{j} -> [{start:.2f}, {end:.2f}], towers {active_towers}")
            mini_interval = {
                "subscript": i,
                "superscript": j,
                "inf": start,
                "sup": end,
                "active_towers": active_towers
            }
            mini_intervals.append(mini_interval)
            j = j + 1

        # print(f"The whole interval can be split as follows")

        for I in interval["interval"]:
            tower = I["tower"]
            if tower not in bfs_nodes:
                continue

            for mini_interval in mini_intervals:
                active_towers = mini_interval["active_towers"]
                subscript = mini_interval["subscript"]
                superscript = mini_interval["superscript"]

                for at in active_towers:
                    if at == tower:
                        I["mini"].append((subscript, superscript))

        for I in interval["interval"]:
            tower = I["tower"]
            if tower not in bfs_nodes:
                continue

            inf = I["inf"]
            sup = I["sup"]
            mini = I["mini"]
            # print(f"  I_{tower} [{inf:.2f}, {sup:.2f}] -> {mini}")

        # Next iteration
        i = i + 1

        # print()

    universe = set()
    collection = []
    tower_ids = []
    for interval in intervals:
        for I in interval["interval"]:
            tower = I["tower"]
            if tower not in bfs_nodes:
                continue

            mini = I["mini"]
            tmp = set()
            for m in mini:
                universe.add(m)
                tmp.add(m)

            idx = tower_ids.index(tower) if tower in tower_ids else -1
            if idx != -1:  # If tower exists in the list
                collection[idx].update(tmp)  # Use update to merge sets
            else:
                collection.append(tmp)  # Append the set directly
                tower_ids.append(tower)

    return universe, collection, tower_ids


def plot_experiment_results(csv_path):
    data = pd.read_csv(csv_path)

    parent_dir = os.path.dirname(os.path.abspath(csv_path))
    img_folder = os.path.join(parent_dir, 'img')
    os.makedirs(img_folder, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(csv_path))[0]

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    axes[0].set_title('Eccentricity over Iterations')
    axes[1].set_title('Total Towers over Iterations')
    axes[2].set_title('Time over Iterations')

    x = data['iteration_seed']

    # Plot eccentricity
    axes[0].plot(x, data['eccentricity_opt'], label='Opt', color='blue', marker='o')
    axes[0].plot(x, data['eccentricity_e_sc_mept'], label='SC MEPT', color='green', marker='*')
    axes[0].plot(x, data['eccentricity_e_t_mept'], label='T MEPT', color='red', marker='s')
    axes[0].set_ylabel('Eccentricity')
    axes[0].legend()
    axes[0].set_xticks(x)
    axes[0].grid(True)

    # Plot total towers
    axes[1].plot(x, data['total_towers_opt'], label='Opt', color='blue', marker='o')
    axes[1].plot(x, data['total_towers_e_sc_mept'], label='SC MEPT', color='green', marker='*')
    axes[1].plot(x, data['total_towers_e_t_mept'], label='T MEPT', color='red', marker='s')
    axes[1].set_ylabel('Total Towers')
    axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].grid(True)

    # Plot time
    axes[2].plot(x, data['time_opt'], label='Opt', color='blue', marker='o')
    axes[2].plot(x, data['time_e_sc_mept'], label='SC MEPT', color='green', marker='*')
    axes[2].plot(x, data['time_e_t_mept'], label='T MEPT', color='red', marker='s')
    axes[2].set_ylabel('Time [s]')
    axes[2].set_xlabel('Iteration Seed')
    axes[2].legend()
    axes[2].set_xticks(x)
    axes[2].grid(True)

    plt.tight_layout()

    img_path = os.path.join(img_folder, f'{base_filename}.png')
    plt.savefig(img_path)
    plt.close()

    # plt.show()

    print(f"Plot saved to {img_path}")

def plot_aggregate(dst, x, towers, times, conf_towers, conf_times):
    opt_towers, sc_towers, t_towers = towers
    opt_times, sc_times, t_times = times

    opt_conf_tower, sc_conf_tower, t_conf_tower = conf_towers
    opt_conf_time, sc_conf_time, t_conf_time = conf_times

    fig, axes = plt.subplots(2, 1, figsize=(10, 15))

    axes[0].set_title('Total Time')
    axes[1].set_title('Total Towers')

    opt_yerr_tower = compute_confidence(opt_conf_tower)
    sc_yerr_tower = compute_confidence(sc_conf_tower)
    t_yerr_tower = compute_confidence(t_conf_tower)

    opt_yerr_time = compute_confidence(opt_conf_time)
    sc_yerr_time = compute_confidence(sc_conf_time)
    t_yerr_time = compute_confidence(t_conf_time)

    # Plot towers with asymmetric error bars
    axes[0].errorbar(x, opt_towers, yerr=opt_yerr_tower, label='Opt', color='blue', marker='o', capsize=5)
    axes[0].errorbar(x, sc_towers, yerr=sc_yerr_tower, label='SC MEPT', color='green', marker='*', capsize=5)
    axes[0].errorbar(x, t_towers, yerr=t_yerr_tower, label='T MEPT', color='red', marker='s', capsize=5)
    axes[0].set_ylabel('Towers')
    axes[0].legend()
    axes[0].set_xticks(x)
    axes[0].grid(True)

    # Plot time with asymmetric error bars
    axes[1].errorbar(x, opt_times, yerr=opt_yerr_time, label='Opt', color='blue', marker='o', capsize=5)
    axes[1].errorbar(x, sc_times, yerr=sc_yerr_time, label='SC MEPT', color='green', marker='*', capsize=5)
    axes[1].errorbar(x, t_times, yerr=t_yerr_time, label='T MEPT', color='red', marker='s', capsize=5)
    axes[1].set_ylabel('Time [s]')
    axes[1].set_xlabel('# Trajectories')
    axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].grid(True)


    plt.tight_layout()

    plt.savefig(dst)
    plt.close()


    print(f"Plot saved to {dst}")
    return

def get_exp_name(scenario, min_rad, max_rad, towers, area, neighbors, star, n_traj, traject_size, dict_sc):
    folder_exp = "exp"
    subfolder_exp = dict_sc[scenario]

    if scenario == 1:
        file_name = f"result_a{area}_t{towers}_r{min_rad}_nt{n_traj}_ts{traject_size}.csv"
        return os.path.join(folder_exp, subfolder_exp, file_name)
    if scenario == 2:
        file_name = f"result_a{area}_t{towers}_rmin{min_rad}_rmax{max_rad}_nt{n_traj}_ts{traject_size}.csv"
        return os.path.join(folder_exp, subfolder_exp, file_name)
    if scenario == 3:
        file_name = f"result_a{area}_t{towers}_nt{n_traj}_ts{traject_size}.csv"
        return os.path.join(folder_exp, subfolder_exp, file_name)
    if scenario == 4:
        file_name = f"result_a{area}_t{towers}_nt{n_traj}_ts{traject_size}.csv"
        return os.path.join(folder_exp, subfolder_exp, file_name)
    if scenario == 5:
        raise "Exception: this scenario not considered."
    if scenario == 6:
        file_name = f"result_a{area}_t{towers}_neig{neighbors}_nt{n_traj}_ts{traject_size}.csv"
        return os.path.join(folder_exp, subfolder_exp, file_name)
    if scenario == 7:
        file_name = f"result_a{area}_t{towers}_star{star}_nt{n_traj}_ts{traject_size}.csv"
        return os.path.join(folder_exp, subfolder_exp, file_name)
    return

def compute_confidence(conf):
    lower = [y[0] for y in conf]
    upper = [y[1] for y in conf]
    return [lower, upper]

def get_confidence(lst):
    return st.t.interval(alpha=0.95, df=len(lst), loc=np.mean(lst), scale=st.sem(lst))
def plot_bars_with_confidence(data, dst):
    # Convert x to a numpy array for easier manipulation
    x = np.array(x)

    # Define the width of the bars
    bar_width = 0.35

    # Set positions for bars (offset them to avoid overlap)
    x_1 = x - bar_width / 2  # Towers will be on the left
    x_2 = x + bar_width / 2  # Times will be on the right

    # Create the figure and two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    ### Subplot 1 - Scenario 1
    axes[0].bar(x_1, towers_scenario1, width=bar_width, yerr=conf_towers_scenario1,
                label='Diagonal Towers', color='blue', capsize=5)
    axes[0].bar(x_times, times_scenario1, width=bar_width, yerr=conf_times_scenario1,
                label=f'Manhattan Towers', color='orange', capsize=5)

    # Set labels and title for the first subplot
    axes[0].set_title(f'Number of Towers')
    axes[0].set_xlabel('#Trajectories')
    axes[0].set_ylabel('Towers')
    axes[0].set_xticks(x)
    axes[0].legend()
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.7)

    ### Subplot 2 - Scenario 2
    axes[1].bar(x_1, towers_scenario2, width=bar_width, yerr=conf_towers_scenario2,
                label=f'{alg_name} Towers', color='blue', capsize=5)
    axes[1].bar(x_times, times_scenario2, width=bar_width, yerr=conf_times_scenario2,
                label=f'{alg_name} Times', color='orange', capsize=5)

    # Set labels and title for the second subplot
    axes[1].set_title(f'{alg_name} - Scenario 2')
    axes[1].set_xlabel('# Trajectories')
    axes[1].set_ylabel('Values')
    axes[1].set_xticks(x)
    axes[1].legend()
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.7)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure
    plt.savefig(dst)
    plt.close()

    print(f"Bar plot with confidence saved to {dst}")