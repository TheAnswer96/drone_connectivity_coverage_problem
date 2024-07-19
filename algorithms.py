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


def multiple_minimum_eccentricity_opt(instance):
    # TODO

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
