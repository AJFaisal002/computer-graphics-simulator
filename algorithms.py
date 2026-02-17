import numpy as np
from collections import deque

# ==========================================================
# LINE & CIRCLE ALGORITHMS
# ==========================================================

def dda_line(x0, y0, x1, y1):
    points = []

    dx = x1 - x0
    dy = y1 - y0

    if dx == 0:
        step = 1 if y1 >= y0 else -1
        for y in range(y0, y1 + step, step):
            points.append((x0, y))
        return points

    m = dy / dx
    x, y = x0, y0
    points.append((round(x), round(y)))

    if abs(m) <= 1:
        step = 1 if x1 >= x0 else -1
        while round(x) != x1:
            x += step
            y += m * step
            points.append((round(x), round(y)))
    else:
        step = 1 if y1 >= y0 else -1
        while round(y) != y1:
            y += step
            x += (1 / m) * step
            points.append((round(x), round(y)))

    return points


def bresenham_line(x0, y0, x1, y1):
    points = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy
    x, y = x0, y0

    while True:
        points.append((x, y))

        if x == x1 and y == y1:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x += sx

        if e2 < dx:
            err += dx
            y += sy

    return points





def midpoint_circle(xc, yc, r):
    points = []

    x = 0
    y = r
    D = 1 - r

    def add_points(x, y):
        return [
            (xc + x, yc + y), (xc - x, yc + y),
            (xc + x, yc - y), (xc - x, yc - y),
            (xc + y, yc + x), (xc - y, yc + x),
            (xc + y, yc - x), (xc - y, yc - x)
        ]

    while x <= y:
        points.extend(add_points(x, y))
        if D < 0:
            D += 2 * x + 1
        else:
            D += 2 * (x - y) + 1
            y -= 1
        x += 1

    return list(dict.fromkeys(points))

def scanline_fill(vertices):
    """
    Scan-Line Polygon Fill Algorithm (Textbook Correct Version)

    Parameters:
        vertices : List of (x, y) polygon vertices (ordered CW or CCW)

    Returns:
        List of filled (x, y) pixel points
    """

    filled_points = []

    n = len(vertices)
    if n < 3:
        return filled_points

    # --------------------------------------
    # Step 1: Build Edge Table (ET)
    # --------------------------------------
    edge_table = {}

    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]

        # Ignore horizontal edges
        if y1 == y2:
            continue

        # Ensure y1 < y2
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        inv_slope = (x2 - x1) / (y2 - y1)

        if y1 not in edge_table:
            edge_table[y1] = []

        edge_table[y1].append({
            "y_max": y2,
            "x": float(x1),
            "inv_slope": inv_slope
        })

    if not edge_table:
        return filled_points

    # --------------------------------------
    # Step 2: Initialize scanline range
    # --------------------------------------
    y_min = min(v[1] for v in vertices)
    y_max = max(v[1] for v in vertices)

    active_edge_table = []
    y = y_min

    # --------------------------------------
    # Step 3: Process each scanline
    # --------------------------------------
    while y <= y_max:

        # Add edges starting at this y
        if y in edge_table:
            active_edge_table.extend(edge_table[y])

        # Remove edges where y == y_max
        active_edge_table = [
            edge for edge in active_edge_table
            if edge["y_max"] > y
        ]

        # Sort AET by current x
        active_edge_table.sort(key=lambda edge: edge["x"])

        # Fill between pairs
        for i in range(0, len(active_edge_table), 2):
            if i + 1 >= len(active_edge_table):
                break

            x_start = int(np.ceil(active_edge_table[i]["x"]))
            x_end   = int(np.floor(active_edge_table[i + 1]["x"]))

            for x in range(x_start, x_end + 1):
                filled_points.append((x, y))

        # Update x for next scanline
        for edge in active_edge_table:
            edge["x"] += edge["inv_slope"]

        y += 1

    return filled_points

def boundary_fill(seed_x, seed_y, boundary_color, fill_color, canvas):
    filled_points = []

    if boundary_color == fill_color:
        return filled_points

    height, width = canvas.shape

    # Seed validation
    if not (0 <= seed_x < width and 0 <= seed_y < height):
        return filled_points

    stack = [(seed_x, seed_y)]

    while stack:
        x, y = stack.pop()

        if not (0 <= x < width and 0 <= y < height):
            continue

        current_color = canvas[y][x]

        if current_color != boundary_color and current_color != fill_color:
            canvas[y][x] = fill_color
            filled_points.append((x, y))

            stack.extend([
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1)
            ])

    return filled_points

def flood_fill(seed_x, seed_y, target_color, replacement_color, canvas):
    filled_points = []

    if target_color == replacement_color:
        return filled_points

    height, width = canvas.shape

    if not (0 <= seed_x < width and 0 <= seed_y < height):
        return filled_points

    if canvas[seed_y][seed_x] != target_color:
        return filled_points

    queue = deque([(seed_x, seed_y)])

    while queue:
        x, y = queue.popleft()

        if not (0 <= x < width and 0 <= y < height):
            continue

        if canvas[y][x] == target_color:
            canvas[y][x] = replacement_color
            filled_points.append((x, y))

            queue.extend([
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1)
            ])

    return filled_points


# ==========================================================
# POLYGON FILLING ALGORITHMS
# ==========================================================

def backface_culling(polygons, view_vector=(0, 0, -1)): 
    """
    Back-Face Culling Algorithm

    Parameters:
        polygons    : List of polygons (each polygon = list of 3D vertices)
        view_vector : Viewing direction vector (default camera along +Z)

    Returns:
        List of visible polygons
    """

    visible_polygons = []

    view_vector = np.array(view_vector)

    for poly in polygons:

        # Ensure polygon has at least 3 vertices
        if len(poly) < 3:
            continue

        v0 = np.array(poly[0])
        v1 = np.array(poly[1])
        v2 = np.array(poly[2])

        # Step 1: Compute surface normal
        normal = np.cross(v1 - v0, v2 - v0)

        # Step 2: Compute dot product
        dot_product = np.dot(normal, view_vector)

        # Step 3: Visibility test
        if dot_product < 0:
            visible_polygons.append(poly)

    return visible_polygons

def edge_function(a, b, c):
    return (c[0] - a[0]) * (b[1] - a[1]) - \
           (c[1] - a[1]) * (b[0] - a[0])
           
def z_buffer(polygons, width, height):

    depth_buffer = np.full((height, width), np.inf)
    color_buffer = np.zeros((height, width))

    for poly in polygons:

        v0 = np.array(poly[0])
        v1 = np.array(poly[1])
        v2 = np.array(poly[2])

        min_x = max(int(min(v0[0], v1[0], v2[0])), 0)
        max_x = min(int(max(v0[0], v1[0], v2[0])), width-1)
        min_y = max(int(min(v0[1], v1[1], v2[1])), 0)
        max_y = min(int(max(v0[1], v1[1], v2[1])), height-1)

        area = edge_function(v0, v1, v2)
        if area == 0:
            continue

        for y in range(min_y, max_y+1):
            for x in range(min_x, max_x+1):

                p = np.array([x, y])

                w0 = edge_function(v1, v2, p)
                w1 = edge_function(v2, v0, p)
                w2 = edge_function(v0, v1, p)

                if (w0 >= 0 and w1 >= 0 and w2 >= 0) or \
                   (w0 <= 0 and w1 <= 0 and w2 <= 0):

                    w0 /= area
                    w1 /= area
                    w2 /= area

                    z = w0*v0[2] + w1*v1[2] + w2*v2[2]

                    if z < depth_buffer[y][x]:
                        depth_buffer[y][x] = z
                        color_buffer[y][x] = 1

    return color_buffer, depth_buffer



def painters_algorithm(polygons):
    """
    Improved Painter's Algorithm
    Depth sorting with basic overlap correction
    """

    # ---- Step 1: Compute average depth ----
    def average_depth(poly):
        return sum(v[2] for v in poly) / len(poly)

    # ---- Step 2: Initial depth sort (far → near) ----
    sorted_polygons = sorted(
        polygons,
        key=average_depth,
        reverse=True
    )

    # ---- Step 3: Basic overlap correction ----
    i = 0
    while i < len(sorted_polygons) - 1:

        p1 = sorted_polygons[i]
        p2 = sorted_polygons[i + 1]

        z1_min = min(v[2] for v in p1)
        z1_max = max(v[2] for v in p1)

        z2_min = min(v[2] for v in p2)
        z2_max = max(v[2] for v in p2)

        # If depth ranges overlap → swap
        if not (z1_max < z2_min or z2_max < z1_min):
            sorted_polygons[i], sorted_polygons[i + 1] = \
                sorted_polygons[i + 1], sorted_polygons[i]

            # restart check from beginning
            i = 0
            continue

        i += 1

    return sorted_polygons


def z_buffer_shaded(polygons, width, height, max_depth):

    depth = np.full((height, width), np.inf)
    color = np.zeros((height, width))

    for poly in polygons:

        if len(poly) < 3:
            continue

        v0 = np.array(poly[0], dtype=float)
        v1 = np.array(poly[1], dtype=float)
        v2 = np.array(poly[2], dtype=float)

        x0, y0 = int(v0[0]), int(v0[1])
        x1, y1 = int(v1[0]), int(v1[1])
        x2, y2 = int(v2[0]), int(v2[1])

        min_x = max(min(x0, x1, x2), 0)
        max_x = min(max(x0, x1, x2), width-1)
        min_y = max(min(y0, y1, y2), 0)
        max_y = min(max(y0, y1, y2), height-1)

        denom = ((y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2))
        if denom == 0:
            continue

        for y in range(min_y, max_y+1):
            for x in range(min_x, max_x+1):

                alpha = ((y1 - y2)*(x - x2) + (x2 - x1)*(y - y2)) / denom
                beta  = ((y2 - y0)*(x - x2) + (x0 - x2)*(y - y2)) / denom
                gamma = 1 - alpha - beta

                if alpha >= 0 and beta >= 0 and gamma >= 0:

                    z = alpha*v0[2] + beta*v1[2] + gamma*v2[2]

                    if z < depth[y, x]:
                        depth[y, x] = z

                        # 🔥 Depth shading (near = white, far = dark)
                        color[y, x] = (max_depth - z) / max_depth

    return color, depth
