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
           
import numpy as np

def z_buffer(polygons, width, height, max_depth=1000):
    """
    Advanced Z-Buffer with Depth Shading
    """

    depth = np.full((height, width), np.inf)
    color = np.zeros((height, width))

    for poly in polygons:

        if len(poly) < 3:
            continue

        v0 = np.array(poly[0], dtype=float)
        v1 = np.array(poly[1], dtype=float)
        v2 = np.array(poly[2], dtype=float)

        # Bounding box
        min_x = max(int(min(v0[0], v1[0], v2[0])), 0)
        max_x = min(int(max(v0[0], v1[0], v2[0])), width - 1)

        min_y = max(int(min(v0[1], v1[1], v2[1])), 0)
        max_y = min(int(max(v0[1], v1[1], v2[1])), height - 1)

        denom = ((v1[1] - v2[1]) * (v0[0] - v2[0]) +
                 (v2[0] - v1[0]) * (v0[1] - v2[1]))

        if denom == 0:
            continue

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):

                # barycentric
                w1 = ((v1[1] - v2[1]) * (x - v2[0]) +
                      (v2[0] - v1[0]) * (y - v2[1])) / denom

                w2 = ((v2[1] - v0[1]) * (x - v2[0]) +
                      (v0[0] - v2[0]) * (y - v2[1])) / denom

                w3 = 1 - w1 - w2

                if w1 >= 0 and w2 >= 0 and w3 >= 0:

                    z = w1*v0[2] + w2*v1[2] + w3*v2[2]

                    if z < depth[y, x]:

                        depth[y, x] = z

                        # 🔥 depth shading
                        intensity = (max_depth - z) / max_depth
                        intensity = np.clip(intensity, 0, 1)

                        color[y, x] = intensity

    return color, depth
import numpy as np

# ================= VECTOR =================
def dot(a, b):
    return np.dot(a, b)

def norm(v):
    return v / (np.linalg.norm(v) + 1e-8)

# ================= NORMAL =================
def normal(a,b,c):
    return norm(np.cross(b-a, c-a))

# ================= SHADING =================
def shade(n, base):
    L = norm(np.array([1,1,-1]))
    diff = max(dot(n, L), 0)

    intensity = 0.6 + 0.8 * diff
    return np.clip(base * 255 * intensity, 0, 255)

# ================= TRIANGLE Z-BUFFER =================
def triangle_zbuffer(img, zbuf, p1, p2, p3, col, W, H):

    d = ((p2[1]-p3[1])*(p1[0]-p3[0]) + (p3[0]-p2[0])*(p1[1]-p3[1]))
    if d == 0:
        return

    minX = int(max(0, min(p1[0],p2[0],p3[0])))
    maxX = int(min(W-1, max(p1[0],p2[0],p3[0])))
    minY = int(max(0, min(p1[1],p2[1],p3[1])))
    maxY = int(min(H-1, max(p1[1],p2[1],p3[1])))

    for y in range(minY, maxY):
        for x in range(minX, maxX):

            w1 = ((p2[1]-p3[1])*(x-p3[0])+(p3[0]-p2[0])*(y-p3[1]))/d
            w2 = ((p3[1]-p1[1])*(x-p3[0])+(p1[0]-p3[0])*(y-p3[1]))/d
            w3 = 1 - w1 - w2

            if w1>=0 and w2>=0 and w3>=0:

                z = w1*p1[2] + w2*p2[2] + w3*p3[2]

                if z < zbuf[y,x]:
                    zbuf[y,x] = z
                    img[y,x] = col
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


import numpy as np

import numpy as np

def z_buffer_shaded(polygons, width, height, max_depth):
    """
    Z-Buffer with Depth Shading (HTML version converted)

    polygons : list of triangles [(x,y,z),...]
    width, height : screen size
    max_depth : used for shading
    """

    depth = np.full((height, width), np.inf)
    color = np.zeros((height, width))

    for poly in polygons:

        if len(poly) < 3:
            continue

        v0 = np.array(poly[0], dtype=float)
        v1 = np.array(poly[1], dtype=float)
        v2 = np.array(poly[2], dtype=float)

        # -------- Bounding Box --------
        min_x = max(int(min(v0[0], v1[0], v2[0])), 0)
        max_x = min(int(max(v0[0], v1[0], v2[0])), width - 1)

        min_y = max(int(min(v0[1], v1[1], v2[1])), 0)
        max_y = min(int(max(v0[1], v1[1], v2[1])), height - 1)

        # -------- Barycentric denominator --------
        denom = ((v1[1] - v2[1]) * (v0[0] - v2[0]) +
                 (v2[0] - v1[0]) * (v0[1] - v2[1]))

        if denom == 0:
            continue

        # -------- Rasterization --------
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):

                # barycentric coordinates
                w1 = ((v1[1] - v2[1]) * (x - v2[0]) +
                      (v2[0] - v1[0]) * (y - v2[1])) / denom

                w2 = ((v2[1] - v0[1]) * (x - v2[0]) +
                      (v0[0] - v2[0]) * (y - v2[1])) / denom

                w3 = 1 - w1 - w2

                # inside triangle
                if w1 >= 0 and w2 >= 0 and w3 >= 0:

                    # depth interpolation
                    z = w1 * v0[2] + w2 * v1[2] + w3 * v2[2]

                    # z-buffer test
                    if z < depth[y, x]:
                        depth[y, x] = z

                        # -------- Depth Shading --------
                        # near = bright, far = dark
                        intensity = (max_depth - z) / max_depth
                        intensity = np.clip(intensity, 0, 1)

                        color[y, x] = intensity

    return color, depth

import numpy as np

# ================= ROTATION =================
def rotate_y(points, angle_deg):
    angle = np.radians(angle_deg)
    cosA = np.cos(angle)
    sinA = np.sin(angle)

    rotated = []
    for x, y, z in points:
        xr = x * cosA + z * sinA
        zr = -x * sinA + z * cosA
        rotated.append((xr, y, zr))
    return rotated


# ================= BACK-FACE CULLING =================
import numpy as np

# ================= ROTATION =================
def rotate_y(points, angle_deg):
    angle = np.radians(angle_deg)
    cosA = np.cos(angle)
    sinA = np.sin(angle)

    rotated = []
    for x, y, z in points:
        xr = x * cosA + z * sinA
        zr = -x * sinA + z * cosA
        rotated.append((xr, y, zr))
    return rotated


# ================= BACK-FACE CULLING =================
import numpy as np

def backface_culling_faces(faces):
    visible = []

    camera = np.array([0, 0, 50.0])   # camera position

    for face in faces:
        v0 = np.array(face[0])
        v1 = np.array(face[1])
        v2 = np.array(face[2])

        # normal (correct order)
        normal = np.cross(v1 - v0, v2 - v0)

        if np.linalg.norm(normal) == 0:
            continue

        normal = normal / np.linalg.norm(normal)

        # face center
        center = (v0 + v1 + v2) / 3.0

        # view vector
        view = center - camera
        view = view / np.linalg.norm(view)

        # visible check
        if np.dot(normal, view) < 0:
            visible.append(face)

    return visible
# ==========================================================
# CLIPPING ALGORITHMS
# ==========================================================

# ----------------------------------------------------------
# Cohen–Sutherland Line Clipping Algorithm
# ----------------------------------------------------------

INSIDE = 0  # 0000
LEFT   = 1  # 0001
RIGHT  = 2  # 0010
BOTTOM = 4  # 0100
TOP    = 8  # 1000

def _compute_code(x, y, x_min, y_min, x_max, y_max):
    code = INSIDE
    if x < x_min:
        code |= LEFT
    elif x > x_max:
        code |= RIGHT
    if y < y_min:
        code |= BOTTOM
    elif y > y_max:
        code |= TOP
    return code


def cohen_sutherland_clip(x0, y0, x1, y1, x_min, y_min, x_max, y_max):
    """
    Cohen–Sutherland Line Clipping Algorithm

    Parameters:
        x0, y0      : Start point of the line
        x1, y1      : End point of the line
        x_min, y_min: Bottom-left corner of the clip rectangle
        x_max, y_max: Top-right corner of the clip rectangle

    Returns:
        (clipped_x0, clipped_y0, clipped_x1, clipped_y1) if line is visible
        None if the line is completely outside the clip window
    """
    code0 = _compute_code(x0, y0, x_min, y_min, x_max, y_max)
    code1 = _compute_code(x1, y1, x_min, y_min, x_max, y_max)

    while True:
        if code0 == INSIDE and code1 == INSIDE:
            return (x0, y0, x1, y1)

        elif code0 & code1:
            return None

        else:
            code_out = code0 if code0 != INSIDE else code1

            if code_out & TOP:
                x = x0 + (x1 - x0) * (y_max - y0) / (y1 - y0)
                y = y_max
            elif code_out & BOTTOM:
                x = x0 + (x1 - x0) * (y_min - y0) / (y1 - y0)
                y = y_min
            elif code_out & RIGHT:
                y = y0 + (y1 - y0) * (x_max - x0) / (x1 - x0)
                x = x_max
            elif code_out & LEFT:
                y = y0 + (y1 - y0) * (x_min - x0) / (x1 - x0)
                x = x_min

            if code_out == code0:
                x0, y0 = x, y
                code0 = _compute_code(x0, y0, x_min, y_min, x_max, y_max)
            else:
                x1, y1 = x, y
                code1 = _compute_code(x1, y1, x_min, y_min, x_max, y_max)


# ----------------------------------------------------------
# Sutherland–Hodgman Polygon Clipping Algorithm
# ----------------------------------------------------------

def _sh_inside(p, edge_start, edge_end):
    return (
        (edge_end[0] - edge_start[0]) * (p[1] - edge_start[1]) -
        (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0])
    ) >= 0


def _sh_intersect(p1, p2, edge_start, edge_end):
    dc = (edge_start[0] - edge_end[0], edge_start[1] - edge_end[1])
    dp = (p1[0] - p2[0], p1[1] - p2[1])
    n1 = edge_start[0] * edge_end[1] - edge_start[1] * edge_end[0]
    n2 = p1[0] * p2[1] - p1[1] * p2[0]
    denom = dc[0] * dp[1] - dc[1] * dp[0]
    if denom == 0:
        return p1
    x = (n1 * dp[0] - n2 * dc[0]) / denom
    y = (n1 * dp[1] - n2 * dc[1]) / denom
    return (x, y)


def sutherland_hodgman_clip(polygon, x_min, y_min, x_max, y_max):
    """
    Sutherland–Hodgman Polygon Clipping Algorithm

    Parameters:
        polygon     : List of (x, y) vertices (ordered CCW or CW)
        x_min, y_min: Bottom-left corner of the clip rectangle
        x_max, y_max: Top-right corner of the clip rectangle

    Returns:
        List of (x, y) vertices of the clipped polygon.
        Empty list if polygon is completely outside.
    """
    clip_edges = [
        ((x_min, y_min), (x_max, y_min)),  # Bottom
        ((x_max, y_min), (x_max, y_max)),  # Right
        ((x_max, y_max), (x_min, y_max)),  # Top
        ((x_min, y_max), (x_min, y_min)),  # Left
    ]

    output = list(polygon)

    for edge_start, edge_end in clip_edges:
        if not output:
            return []
        input_list = output
        output = []
        for i in range(len(input_list)):
            current = input_list[i]
            previous = input_list[i - 1]
            if _sh_inside(current, edge_start, edge_end):
                if not _sh_inside(previous, edge_start, edge_end):
                    output.append(_sh_intersect(previous, current, edge_start, edge_end))
                output.append(current)
            elif _sh_inside(previous, edge_start, edge_end):
                output.append(_sh_intersect(previous, current, edge_start, edge_end))

    return output
