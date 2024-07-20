from util import is_zero


def single_minimum_eccentricity(instance):
    # Result of the random instance
    G = instance["graph"]
    intervals = instance["intervals"]
    print("There are %d trajectories" % len(intervals))
    for interval in intervals:
        length = interval["length"]
        print(" Length=%.2f" % length)
        for I in interval["interval"]:
            tower = I["tower"]
            inf = I["inf"]
            sup = I["sup"]
            print("  T%d [%.2f, %.2f]" % (tower, inf, sup))

    # TODO

    output = {
        "eccentricity": -1,
        "used_intervals": [1, 2, 3],
    }

    return output


def single_minimum_coverage(instance):
    # TODO

    output = {
        "result": -1
    }

    return output


def single_minimum_k_coverage(instance):
    # TODO

    output = {
        "result": -1
    }

    return output


def create_instance_set_cover(intervals):
    i = 0
    for interval in intervals:
        print(f"Trajectory_{i}")
        length = interval["length"]
        print(f"Length={length:.2f}")
        endpoints = []
        for I in interval["interval"]:
            tower = I["tower"]
            inf = I["inf"]
            sup = I["sup"]
            endpoints.append((inf, 'start', tower))
            endpoints.append((sup, 'end', tower))
            print(f"  I_{tower} [{inf:.2f}, {sup:.2f}]")

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
        for start, end, active_towers in segments:
            print(f"I_{i}^{j} -> [{start:.2f}, {end:.2f}]: Towers {active_towers}")
            mini_interval = {
                "subscript": i,
                "superscript": j,
                "inf": start,
                "sup": end,
                "active_towers": active_towers
            }
            mini_intervals.append(mini_interval)
            j = j + 1

        for I in interval["interval"]:
            tower = I["tower"]

            for mini_interval in mini_intervals:
                active_towers = mini_interval["active_towers"]
                subscript = mini_interval["subscript"]
                superscript = mini_interval["superscript"]

                for at in active_towers:
                    if at == tower:
                        I["mini"].append((subscript, superscript))

        for I in interval["interval"]:
            tower = I["tower"]
            inf = I["inf"]
            sup = I["sup"]
            mini = I["mini"]
            print(f"  I_{tower} [{inf:.2f}, {sup:.2f}] -> ")
            for subscript, superscript in mini:
                print(f"      I_{subscript}^{superscript}")

        # Next iteration
        i = i + 1

        print()

    return intervals


def multiple_minimum_eccentricity_opt(instance):
    G = instance["graph"]
    intervals = create_instance_set_cover(instance["intervals"])

    output = {
        "result": -1
    }

    return output


def multiple_minimum_eccentricity_v1(instance):
    # TODO

    output = {
        "result": -1
    }

    return output


def multiple_minimum_eccentricity_v2(instance):
    # TODO

    output = {
        "result": -1
    }

    return output
