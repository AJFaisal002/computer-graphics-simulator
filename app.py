import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from algorithms import *

# ===================== ALGORITHMS =========================

def dda_line(x0, y0, x1, y1):
    points = []
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))

    if steps == 0:
        return [(x0, y0)]

    x_inc = dx / steps
    y_inc = dy / steps

    x, y = x0, y0
    for _ in range(steps + 1):
        points.append((round(x), round(y)))
        x += x_inc
        y += y_inc

    return points


def midpoint_circle(xc, yc, r):
    points = []
    x = 0
    y = r
    p = 1 - r

    while x <= y:
        pts = [
            (xc + x, yc + y), (xc - x, yc + y),
            (xc + x, yc - y), (xc - x, yc - y),
            (xc + y, yc + x), (xc - y, yc + x),
            (xc + y, yc - x), (xc - y, yc - x)
        ]
        points.extend(pts)

        if p < 0:
            p += 2*x + 3
        else:
            p += 2*(x - y) + 5
            y -= 1
        x += 1

    return list(set(points))


def scanline_fill(vertices):
    filled = []
    n = len(vertices)
    if n < 3:
        return filled

    edge_table = {}

    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i+1)%n]

        if y1 == y2:
            continue

        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        inv_slope = (x2 - x1) / (y2 - y1)

        edge_table.setdefault(y1, []).append({
            "y_max": y2,
            "x": float(x1),
            "inv_slope": inv_slope
        })

    if not edge_table:
        return filled

    y = min(edge_table.keys())
    y_max = max(e["y_max"] for edges in edge_table.values() for e in edges)
    AET = []

    while y < y_max:
        if y in edge_table:
            AET.extend(edge_table[y])

        AET = [e for e in AET if e["y_max"] > y]
        AET.sort(key=lambda e: e["x"])

        for i in range(0, len(AET)-1, 2):
            x_start = int(np.ceil(AET[i]["x"]))
            x_end = int(np.floor(AET[i+1]["x"]))
            for x in range(x_start, x_end+1):
                filled.append((x, y))

        for e in AET:
            e["x"] += e["inv_slope"]

        y += 1

    return filled


def boundary_fill(seed_x, seed_y, boundary_color, fill_color, canvas):
    stack = [(seed_x, seed_y)]
    h, w = canvas.shape

    while stack:
        x, y = stack.pop()
        if 0 <= x < w and 0 <= y < h:
            if canvas[y][x] != boundary_color and canvas[y][x] != fill_color:
                canvas[y][x] = fill_color
                stack.extend([(x+1,y),(x-1,y),(x,y+1),(x,y-1)])


def flood_fill(seed_x, seed_y, target, replacement, canvas):
    queue = deque([(seed_x, seed_y)])
    h, w = canvas.shape

    while queue:
        x, y = queue.popleft()
        if 0 <= x < w and 0 <= y < h:
            if canvas[y][x] == target:
                canvas[y][x] = replacement
                queue.extend([(x+1,y),(x-1,y),(x,y+1),(x,y-1)])


# ===================== 3D =======================

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

def project(points, scale=60, offset=150):
    projected = []
    for x, y, z in points:
        xp = x * scale + offset
        yp = y * scale + offset
        projected.append((xp, yp, z))
    return projected


def backface_culling(polygons, view=(0,0,-1)):
    visible = []
    view = np.array(view)

    for poly in polygons:
        v0 = np.array(poly[0])
        v1 = np.array(poly[1])
        v2 = np.array(poly[2])
        normal = np.cross(v1-v0, v2-v0)
        if np.dot(normal, view) < 0:
            visible.append(poly)
    return visible


def painters_algorithm(polygons):
    return sorted(polygons,
                  key=lambda p: sum(v[2] for v in p)/len(p),
                  reverse=True)


def z_buffer(polygons, width, height):
    depth = np.full((height, width), np.inf)
    color = np.zeros((height, width))

    for poly in polygons:
        for v in poly:
            x, y, z = int(v[0]), int(v[1]), v[2]
            if 0 <= x < width and 0 <= y < height:
                if z < depth[y][x]:
                    depth[y][x] = z
                    color[y][x] = 1

    return color, depth


# ==========================================================
# ================= STREAMLIT APP ==========================
# ==========================================================

st.set_page_config(layout="wide")
st.title("🖥️ Computer Graphics Algorithms Simulator")

st.sidebar.header("Controls")

algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["DDA Line","Bresenham Line","Midpoint Circle",
     "Scan-Line Fill","Boundary Fill","Flood Fill",
     "Back-Face Culling","Painter's Algorithm","Z-Buffer",
     "Cohen-Sutherland Clipping","Sutherland-Hodgman Clipping"]
)

# ================= INPUT CONTROLS =================

if algorithm in ["DDA Line", "Bresenham Line"]:

    st.sidebar.subheader("Line Points")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        x0 = st.number_input("x0", value=2)
    with col2:
        y0 = st.number_input("y0", value=3)

    col3, col4 = st.sidebar.columns(2)
    with col3:
        x1 = st.number_input("x1", value=20)
    with col4:
        y1 = st.number_input("y1", value=15)

elif algorithm == "Midpoint Circle":

    st.sidebar.subheader("Circle Parameters")

    col1, col2, col3 = st.sidebar.columns(3)

    with col1:
        xc = st.number_input("Center X", value=0)
    with col2:
        yc = st.number_input("Center Y", value=0)
    with col3:
        r = st.number_input("Radius", min_value=1, value=20)

elif algorithm in ["Scan-Line Fill","Boundary Fill","Flood Fill"]:

    st.sidebar.subheader("Polygon Vertices")
    num_vertices = st.sidebar.number_input("Number of vertices", 3, 8, 4)

    vertices = []
    default_rect = [(50,50),(150,50),(150,120),(50,120)]

    for i in range(num_vertices):
        if num_vertices == 4:
            dx, dy = default_rect[i]
        else:
            angle = 2*np.pi*i/num_vertices
            dx = int(150 + 80*np.cos(angle))
            dy = int(150 + 80*np.sin(angle))

        col1, col2 = st.sidebar.columns(2)
        with col1:
            x = st.number_input(f"x{i+1}", value=dx, key=f"x{i}")
        with col2:
            y = st.number_input(f"y{i+1}", value=dy, key=f"y{i}")
        vertices.append((int(x), int(y)))

    seed_x = int(sum(v[0] for v in vertices)/len(vertices))
    seed_y = int(sum(v[1] for v in vertices)/len(vertices))

elif algorithm in ["Back-Face Culling","Painter's Algorithm","Z-Buffer"]:
    size = st.sidebar.slider("Cube Size",1,10,5)
    angle = st.sidebar.slider("Rotate Y (degrees)",0,360,45)

elif algorithm in ["Cohen-Sutherland Clipping","Sutherland-Hodgman Clipping"]:

    st.sidebar.subheader("Clip Window")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x_min = st.number_input("x_min", value=50)
        y_min = st.number_input("y_min", value=50)
    with col2:
        x_max = st.number_input("x_max", value=200)
        y_max = st.number_input("y_max", value=200)

    if algorithm == "Cohen-Sutherland Clipping":
        st.sidebar.subheader("Line Points")
        col3, col4 = st.sidebar.columns(2)
        with col3:
            lx0 = st.number_input("x0", value=20)
            ly0 = st.number_input("y0", value=100)
        with col4:
            lx1 = st.number_input("x1", value=250)
            ly1 = st.number_input("y1", value=180)

    else:
        st.sidebar.subheader("Polygon Vertices")
        num_clip_verts = st.sidebar.number_input("Number of vertices", 3, 8, 5)
        clip_vertices = []
        default_poly = [(30,90),(120,20),(230,60),(210,210),(60,200)]
        for i in range(num_clip_verts):
            if num_clip_verts == 5 and i < 5:
                dvx, dvy = default_poly[i]
            else:
                angle_c = 2*np.pi*i/num_clip_verts
                dvx = int(130 + 100*np.cos(angle_c))
                dvy = int(130 + 100*np.sin(angle_c))
            col5, col6 = st.sidebar.columns(2)
            with col5:
                vx = st.number_input(f"x{i+1}", value=dvx, key=f"cvx{i}")
            with col6:
                vy = st.number_input(f"y{i+1}", value=dvy, key=f"cvy{i}")
            clip_vertices.append((int(vx), int(vy)))

draw = st.sidebar.button("▶ Draw")

# ==========================================================
# DRAW SECTION
# ==========================================================

if draw:

    fig = plt.figure(figsize=(6,6))

    # ================= 2D =================
    if algorithm in ["DDA Line","Bresenham Line","Midpoint Circle",
                     "Scan-Line Fill","Boundary Fill","Flood Fill"]:

        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        ax.grid(True)

        if algorithm == "DDA Line":
            pts = dda_line(x0,y0,x1,y1)
            xs, ys = zip(*pts)
            ax.scatter(xs, ys)

        elif algorithm == "Bresenham Line":
            pts = bresenham_line(x0, y0, x1, y1)
            xs, ys = zip(*pts)
            ax.scatter(xs, ys)

        elif algorithm == "Midpoint Circle":
            pts = midpoint_circle(xc,yc,r)
            xs, ys = zip(*pts)
            ax.scatter(xs, ys)

        else:
            # draw polygon boundary
            for i in range(len(vertices)):
                edge = dda_line(*vertices[i], *vertices[(i+1)%len(vertices)])
                xs, ys = zip(*edge)
                ax.plot(xs, ys)

            if algorithm == "Scan-Line Fill":
                pts = scanline_fill(vertices)
                if pts:
                    xs, ys = zip(*pts)
                    ax.scatter(xs, ys, s=25, marker='s')

            else:
                canvas = np.zeros((300,300))
                for i in range(len(vertices)):
                    edge = dda_line(*vertices[i], *vertices[(i+1)%len(vertices)])
                    for px, py in edge:
                        if 0 <= px < 300 and 0 <= py < 300:
                            canvas[py][px] = 1
                            if px+1 < 300: canvas[py][px+1] = 1
                            if py+1 < 300: canvas[py+1][px] = 1

                if algorithm == "Boundary Fill":
                    boundary_fill(seed_x, seed_y, 1, 2, canvas)
                    fy, fx = np.where(canvas == 2)
                else:
                    flood_fill(seed_x, seed_y, 0, 3, canvas)
                    fy, fx = np.where(canvas == 3)

                ax.scatter(fx, fy, s=5)

        st.pyplot(fig)

    # ================= CLIPPING =================
    elif algorithm == "Cohen-Sutherland Clipping":

        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 300)
        ax.set_title("Cohen-Sutherland Line Clipping")

        rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                              linewidth=2, edgecolor="black",
                              facecolor="lightyellow", label="Clip Window")
        ax.add_patch(rect)

        ax.plot([lx0, lx1], [ly0, ly1], color="lightblue", linewidth=2,
                linestyle="--", label="Original Line")

        result = cohen_sutherland_clip(lx0, ly0, lx1, ly1,
                                       x_min, y_min, x_max, y_max)
        if result:
            cx0, cy0, cx1, cy1 = result
            ax.plot([cx0, cx1], [cy0, cy1], color="red",
                    linewidth=3, label="Clipped Line")
            ax.scatter([cx0, cx1], [cy0, cy1], color="red", zorder=5)
            st.success(f"Clipped Line: ({cx0:.1f}, {cy0:.1f}) -> ({cx1:.1f}, {cy1:.1f})")
        else:
            st.warning("Line is completely outside the clip window.")

        ax.legend()
        st.pyplot(fig)

    elif algorithm == "Sutherland-Hodgman Clipping":

        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 300)
        ax.set_title("Sutherland-Hodgman Polygon Clipping")

        rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                              linewidth=2, edgecolor="black",
                              facecolor="lightyellow", label="Clip Window")
        ax.add_patch(rect)

        orig_xs = [v[0] for v in clip_vertices] + [clip_vertices[0][0]]
        orig_ys = [v[1] for v in clip_vertices] + [clip_vertices[0][1]]
        ax.plot(orig_xs, orig_ys, color="lightblue", linewidth=2,
                linestyle="--", label="Original Polygon")

        clipped = sutherland_hodgman_clip(clip_vertices,
                                          x_min, y_min, x_max, y_max)
        if clipped:
            clip_xs = [v[0] for v in clipped] + [clipped[0][0]]
            clip_ys = [v[1] for v in clipped] + [clipped[0][1]]
            ax.fill(clip_xs, clip_ys, alpha=0.4, color="red", label="Clipped Region")
            ax.plot(clip_xs, clip_ys, color="red", linewidth=2)
            st.success(f"Clipped polygon has {len(clipped)} vertices.")
        else:
            st.warning("Polygon is completely outside the clip window.")

        ax.legend()
        st.pyplot(fig)

    # ================= 3D =================
    else:
        ax = fig.add_subplot(111, projection='3d')

        cube = [
            [(0,0,0),(size,0,0),(size,size,0)],
            [(0,0,0),(size,size,0),(0,size,0)],
            [(0,0,size),(size,0,size),(size,size,size)],
            [(0,0,size),(size,size,size),(0,size,size)]
        ]

        cube = [rotate_y(p, angle) for p in cube]

        if algorithm == "Back-Face Culling":
            cube = backface_culling(cube)
        elif algorithm == "Painter's Algorithm":
            cube = painters_algorithm(cube)
        elif algorithm == "Z-Buffer":

            cube = [
                # Front
                [(0,0,0),(size,0,0),(size,size,0)],
                [(0,0,0),(size,size,0),(0,size,0)],
                # Back
                [(0,0,size),(size,0,size),(size,size,size)],
                [(0,0,size),(size,size,size),(0,size,size)],
                # Left
                [(0,0,0),(0,size,0),(0,size,size)],
                [(0,0,0),(0,size,size),(0,0,size)],
                # Right
                [(size,0,0),(size,size,0),(size,size,size)],
                [(size,0,0),(size,size,size),(size,0,size)],
                # Top
                [(0,size,0),(size,size,0),(size,size,size)],
                [(0,size,0),(size,size,size),(0,size,size)],
                # Bottom
                [(0,0,0),(size,0,0),(size,0,size)],
                [(0,0,0),(size,0,size),(0,0,size)],
            ]

            cube = [rotate_y(p, angle) for p in cube]
            cube = [[(x, y, z + size + 3) for (x, y, z) in poly] for poly in cube]

            scale = 50
            offset = 120
            d = 5

            projected = []
            for poly in cube:
                new_poly = []
                for x, y, z in poly:
                    factor = d / (z + d + 1e-5)
                    xp = x * factor
                    yp = y * factor
                    sx = xp * scale + offset
                    sy = yp * scale + offset
                    new_poly.append((sx, sy, z))
                projected.append(new_poly)

            color, depth = z_buffer_shaded(projected, 300, 300, size)

            fig2 = plt.figure(figsize=(6,6))
            plt.imshow(color, cmap="gray")
            plt.axis("off")

            st.pyplot(fig2)
            st.stop()

        for poly in cube:
            xs = [v[0] for v in poly] + [poly[0][0]]
            ys = [v[1] for v in poly] + [poly[0][1]]
            zs = [v[2] for v in poly] + [poly[0][2]]
            ax.plot(xs,ys,zs)

        st.pyplot(fig)

st.markdown("---")
st.caption("Computer Graphics Algorithms Simulator • Fully Interactive Version")