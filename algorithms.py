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


def create_instance_set_cover(instance):
    G = instance["graph"]
    intervals = instance["intervals"]
    i = 0
    for interval in intervals:
        length = interval["length"]
        print()
        endpoints = []
        for I in interval["interval"]:
            tower = I["tower"]
            inf = I["inf"]
            sup = I["sup"]
            endpoints.append((inf, 'start', tower))
            endpoints.append((sup, 'end', tower))
            # print("  T%d [%.2f, %.2f]" % (tower, inf, sup))

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

        i = i+1
        for start, end, active_towers in segments:
            print(f"  [{start:.2f}, {end:.2f}]: Towers {active_towers}")



def multiple_minimum_eccentricity_opt(instance):
    new_instance = create_instance_set_cover(instance)

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
