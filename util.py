import math


def is_square(n):
    if n < 0:
        return False

    root = math.isqrt(n)
    return root * root == n


def get_distance(p0, p1):
    return math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
