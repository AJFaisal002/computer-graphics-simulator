import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# ==========================================================
# 1️⃣ DDA Line
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
            x += (1/m) * step
            points.append((round(x), round(y)))

    return points


# ==========================================================
# 2️⃣ Bresenham
# ==========================================================

def bresenham_line(x0, y0, x1, y1):
    points = []

    dx = x1 - x0
    dy = y1 - y0

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    sx = 1 if dx >= 0 else -1
    sy = 1 if dy >= 0 else -1

    if abs_dx >= abs_dy:
        p = 2*abs_dy - abs_dx
        x, y = x0, y0
        for _ in range(abs_dx + 1):
            points.append((x,y))
            if p < 0:
                p += 2*abs_dy
            else:
                y += sy
                p += 2*abs_dy - 2*abs_dx
            x += sx
    else:
        p = 2*abs_dx - abs_dy
        x, y = x0, y0
        for _ in range(abs_dy + 1):
            points.append((x,y))
            if p < 0:
                p += 2*abs_dx
            else:
                x += sx
                p += 2*abs_dx - 2*abs_dy
            y += sy

    return points


# ==========================================================
# 3️⃣ Midpoint Circle
# ==========================================================

def midpoint_circle(xc, yc, r):
    points = []
    x = 0
    y = r
    D = 1 - r

    def add_points(x,y):
        return [
            (xc+x,yc+y),(xc-x,yc+y),
            (xc+x,yc-y),(xc-x,yc-y),
            (xc+y,yc+x),(xc-y,yc+x),
            (xc+y,yc-x),(xc-y,yc-x)
        ]

    while x <= y:
        points.extend(add_points(x,y))
        if D < 0:
            D += 2*x + 1
        else:
            D += 2*(x-y) + 1
            y -= 1
        x += 1

    return list(dict.fromkeys(points))

def scanline_fill(vertices):
    filled = []
    edge_table = {}
    n = len(vertices)

    # Build Edge Table
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i+1) % n]

        if y1 == y2:
            continue

        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        inv_slope = (x2 - x1) / (y2 - y1)

        if y1 not in edge_table:
            edge_table[y1] = []

        edge_table[y1].append({
            "y_max": y2,
            "x": x1,
            "inv_slope": inv_slope
        })

    y = min(edge_table.keys())
    y_max = max(v[1] for v in vertices)
    active = []

    while y <= y_max:

        if y in edge_table:
            active.extend(edge_table[y])

        active = [e for e in active if e["y_max"] > y]
        active.sort(key=lambda e: e["x"])

        for i in range(0, len(active), 2):
            if i+1 < len(active):
                x_start = int(np.ceil(active[i]["x"]))
                x_end   = int(np.floor(active[i+1]["x"]))
                for x in range(x_start, x_end+1):
                    filled.append((x,y))

        for e in active:
            e["x"] += e["inv_slope"]

        y += 1

    return filled

def boundary_fill(sx, sy, boundary, fill, canvas):
    filled = []
    height, width = canvas.shape

    stack = [(sx, sy)]

    while stack:
        x, y = stack.pop()

        if not (0 <= x < width and 0 <= y < height):
            continue

        if canvas[y][x] != boundary and canvas[y][x] != fill:
            canvas[y][x] = fill
            filled.append((x,y))

            stack.extend([
                (x+1,y), (x-1,y),
                (x,y+1), (x,y-1)
            ])

    return filled


def flood_fill(sx, sy, target, new, canvas):
    filled = []
    height, width = canvas.shape

    if target == new:
        return filled

    if not (0 <= sx < width and 0 <= sy < height):
        return filled

    if canvas[sy][sx] != target:
        return filled

    queue = deque([(sx,sy)])

    while queue:
        x,y = queue.popleft()

        if not (0 <= x < width and 0 <= y < height):
            continue

        if canvas[y][x] == target:
            canvas[y][x] = new
            filled.append((x,y))

            queue.extend([
                (x+1,y),(x-1,y),
                (x,y+1),(x,y-1)
            ])

    return filled

def backface_culling(polygons, view=(0,0,1)):
    visible = []
    view = np.array(view)

    for poly in polygons:
        v0 = np.array(poly[0])
        v1 = np.array(poly[1])
        v2 = np.array(poly[2])

        normal = np.cross(v1 - v0, v2 - v0)

        if np.dot(normal, view) > 0:   # CHANGED SIGN
            visible.append(poly)

    return visible



def painters_algorithm(polygons):
    return sorted(
        polygons,
        key=lambda p: max(v[2] for v in p),
        reverse=True
    )

def edge_function(a,b,c):
    return (c[0]-a[0])*(b[1]-a[1]) - (c[1]-a[1])*(b[0]-a[0])


def z_buffer(polygons, width=200, height=200):

    depth = np.full((height,width), np.inf)
    color = np.zeros((height,width))

    for poly in polygons:

        if len(poly) < 3:
            continue

        v0 = np.array(poly[0])
        v1 = np.array(poly[1])
        v2 = np.array(poly[2])

        min_x = max(int(min(v0[0],v1[0],v2[0])),0)
        max_x = min(int(max(v0[0],v1[0],v2[0])),width-1)
        min_y = max(int(min(v0[1],v1[1],v2[1])),0)
        max_y = min(int(max(v0[1],v1[1],v2[1])),height-1)

        area = edge_function(v0,v1,v2)
        if area == 0:
            continue

        for y in range(min_y, max_y+1):
            for x in range(min_x, max_x+1):

                p = np.array([x,y])

                w0 = edge_function(v1,v2,p)
                w1 = edge_function(v2,v0,p)
                w2 = edge_function(v0,v1,p)

                if w0 >= 0 and w1 >= 0 and w2 >= 0:

                    w0 /= area
                    w1 /= area
                    w2 /= area

                    z = w0*v0[2] + w1*v1[2] + w2*v2[2]

                    if z < depth[y][x]:
                        depth[y][x] = z
                        color[y][x] = 1

    return color
