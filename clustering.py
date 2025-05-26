# Implements clustering algorithm

from collections import deque, defaultdict
import numpy as np

from prep_processing import ParkingSpotAlter


def angle_diff_check(k1, k2):
    """Calculates the minimal angle difference between two line slopes.

    Returns:
        float: Minimal angle difference in radians [0, π/2]
    """
    angle1 = np.arctan(k1)
    angle2 = np.arctan(k2)
    d = abs(angle2 - angle1)
    return min(d, np.pi - d)


def are_adjacent(a: ParkingSpotAlter, b: ParkingSpotAlter, a_line, a_sidelength, tol_coef):
    """Checks if two parking spots are geometrically adjacent.

    Args:
        a: First parking spot
        b: Second parking spot
        a_line: Reference line parameters (slope, intercept)
        a_sidelength: Length of reference side
        tol_coef: Tolerance coefficient for distance check

    Returns:
        bool: True if spots are adjacent
    """
    p1 = np.array([a.ps.center])
    p2 = np.array([b.ps.center])

    real_length = np.linalg.norm(p1 - p2)

    k_h = angle_diff_check(a_line[0], b.h_line[0])
    k_w = angle_diff_check(a_line[0], b.w_line[0])

    if k_h < k_w:
        approx_length = a_sidelength / 2 + b.height / 2
    else:
        approx_length = a_sidelength / 2 + b.width / 2

    tolerance = tol_coef * approx_length

    if abs(real_length - approx_length) < tolerance:
        return True
    return False


def get_slope(k0, x1, y1, x2, y2):
    """Verifies if line segment matches reference slope within tolerance.

    Args:
        k0: Reference slope
        x1,y1: Start point coordinates
        x2,y2: End point coordinates
    """
    if x2 == x1:
        return float("inf")
    k = (y2 - y1) / (x2 - x1)
    return angle_diff_check(k0, k) < np.pi / 10


def assign_clusters(spots, tolerance_coef):
    """Groups parking spots into clusters using BFS-based algorithm."""
    visited = [False] * len(spots)
    cluster_id = 0

    def bfs(index):
        queue = deque([index])
        while queue:
            current = queue.popleft()

            if visited[current]:
                continue

            visited[current] = True
            spots[current].ps.cluster = cluster_id

            # Vertical
            num_el = 0
            for neighbor_index, other in enumerate(spots):
                if num_el == 2:
                    break
                if not visited[neighbor_index]:
                    if get_slope(
                        spots[current].h_line[0],
                        spots[current].ps.center[0],
                        spots[current].ps.center[1],
                        other.ps.center[0],
                        other.ps.center[1],
                    ):
                        if are_adjacent(
                            spots[current],
                            other,
                            spots[current].h_line,
                            spots[current].height,
                            tolerance_coef,
                        ):
                            queue.append(neighbor_index)
                            num_el += 1

            # Horizontal
            num_el = 0
            for neighbor_index, other in enumerate(spots):
                if num_el == 2:
                    break
                if not visited[neighbor_index]:
                    if get_slope(
                        spots[current].w_line[0],
                        spots[current].ps.center[0],
                        spots[current].ps.center[1],
                        other.ps.center[0],
                        other.ps.center[1],
                    ):
                        if are_adjacent(
                            spots[current],
                            other,
                            spots[current].w_line,
                            spots[current].width,
                            tolerance_coef,
                        ):
                            queue.append(neighbor_index)
                            num_el += 1

    for i in range(len(spots)):
        if not visited[i]:
            bfs(i)
            cluster_id += 1

    cluster_counts = defaultdict(list)

    for i, spot in enumerate(spots):
        cid = spot.ps.cluster
        if cid != -1:
            cluster_counts[cid].append(i)

    for cid, indices in cluster_counts.items():
        if len(indices) < 3:
            for i in indices:
                spots[i].ps.cluster = -1
