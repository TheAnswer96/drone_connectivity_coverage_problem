import math

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
