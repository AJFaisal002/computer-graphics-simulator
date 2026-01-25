# algorithms.py

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
    x = x0
    y = y0
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

    dx = x1 - x0
    dy = y1 - y0

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    sx = 1 if dx >= 0 else -1
    sy = 1 if dy >= 0 else -1

    if abs_dx >= abs_dy:
        p = 2 * abs_dy - abs_dx
        x, y = x0, y0

        for _ in range(abs_dx + 1):
            points.append((x, y))
            if p < 0:
                p += 2 * abs_dy
            else:
                y += sy
                p += 2 * abs_dy - 2 * abs_dx
            x += sx
    else:
        p = 2 * abs_dx - abs_dy
        x, y = x0, y0

        for _ in range(abs_dy + 1):
            points.append((x, y))
            if p < 0:
                p += 2 * abs_dx
            else:
                x += sx
                p += 2 * abs_dx - 2 * abs_dy
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
