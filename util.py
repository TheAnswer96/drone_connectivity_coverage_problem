import math

from algorithms import *

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
