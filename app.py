import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import streamlit as st
import numpy as np
import time
from algorithms import *
from algorithms import rotate_y, backface_culling_faces
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from algorithms import triangle_zbuffer, normal, shade

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

        if not (0 <= x < w and 0 <= y < h):
            continue

        # 🔥 STRICT CHECK (VERY IMPORTANT)
        if canvas[y][x] == boundary_color or canvas[y][x] == fill_color:
            continue

        canvas[y][x] = fill_color

        stack.append((x+1,y))
        stack.append((x-1,y))
        stack.append((x,y+1))
        stack.append((x,y-1))

def flood_fill(seed_x, seed_y, target, replacement, canvas):
    if target == replacement:
        return   # 🔥 important

    queue = deque([(seed_x, seed_y)])
    h, w = canvas.shape

    while queue:
        x, y = queue.popleft()

        if 0 <= x < w and 0 <= y < h:

            # 🔥 STRICT CHECK
            if canvas[y][x] != target:
                continue

            canvas[y][x] = replacement

            queue.append((x+1,y))
            queue.append((x-1,y))
            queue.append((x,y+1))
            queue.append((x,y-1))
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


def backface_culling_faces(faces):
    visible = []

    camera = np.array([0, 0, 100])  # 🔥 camera in front

    for face in faces:
        v0 = np.array(face[0])
        v1 = np.array(face[1])
        v2 = np.array(face[2])

        # normal
        normal = np.cross(v1 - v0, v2 - v0)
        if np.linalg.norm(normal) == 0:
            continue
        normal = normal / np.linalg.norm(normal)

        # face center
        center = (v0 + v1 + v2) / 3

        # view vector (IMPORTANT)
        view = camera - center
        view = view / np.linalg.norm(view)

        # 🔥 CORRECT CONDITION
        if np.dot(normal, view) > 0:
            visible.append(face)

    return visible


def painters_algorithm(polygons):
    return sorted(polygons,
                  key=lambda p: sum(v[2] for v in p)/len(p),
                  reverse=True)



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
        # ================= Z-BUFFER (FIXED 🔥) =================
# ================= Z-BUFFER (FIXED 🔥) =================
if "draw" not in st.session_state:
    st.session_state.draw = False

if st.sidebar.button("▶ Draw"):
    st.session_state.draw = True
    
if "run_z" not in st.session_state:
    st.session_state.run_z = False

if st.session_state.draw and algorithm == "Z-Buffer":
    st.session_state.run_z = True            


# ==========================================================
# DRAW SECTION
# ==========================================================

if st.session_state.draw and algorithm != "Z-Buffer":
    

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
                            for dx in [-1,0,1]:
                                for dy in [-1,0,1]:
                                    nx, ny = px + dx, py + dy
                                    if 0 <= nx < 300 and 0 <= ny < 300:
                                        canvas[ny][nx] = 1

                if algorithm == "Boundary Fill":
                    boundary_fill(seed_x, seed_y, 1, 2, canvas)
                    fy, fx = np.where(canvas == 2)
                else:
                    flood_fill(seed_x, seed_y, 0, 3, canvas)
                    fy, fx = np.where(canvas == 3)

                ax.scatter(fx, fy, s=5)

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
    elif algorithm == "Cohen-Sutherland Clipping":

        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 300)
        ax.set_title("Cohen-Sutherland Line Clipping")



        # ===== DRAW CLIP WINDOW =====
        rect = plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor="black",
            facecolor="lightyellow",
            label="Clip Window"
        )
        ax.add_patch(rect)

        # ===== ORIGINAL LINE =====
        ax.plot(
            [lx0, lx1],
            [ly0, ly1],
            linestyle="--",
            color="skyblue",
            linewidth=2,
            label="Original Line"
        )

        # ===== APPLY CLIPPING =====
        clipped = cohen_sutherland_clip(
            lx0, ly0, lx1, ly1,
            x_min, y_min, x_max, y_max
        )

        # ===== RESULT TEXT (TOP GREEN BOX) =====
        if clipped:
            x0c, y0c, x1c, y1c = clipped

            st.success(
                f"Clipped Line: ({x0c:.1f}, {y0c:.1f}) → ({x1c:.1f}, {y1c:.1f})"
            )

            # ===== DRAW CLIPPED LINE =====
            ax.plot(
                [x0c, x1c],
                [y0c, y1c],
                color="red",
                linewidth=3,
                label="Clipped Line"
            )

        else:
            st.error("Line is completely outside the clipping window ❌")

        # ===== LEGEND =====
        ax.legend()

        # ===== SHOW =====
        st.pyplot(fig)    

    # ================= 3D =================
# ================= 3D =================
    else:
        ax = fig.add_subplot(111, projection='3d')

        # ================= BACK-FACE ONLY FIX =================
        if algorithm == "Back-Face Culling":

    # ===== VERTICES =====
            verts = np.array([
                [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]
            ]) * size

            # ===== ROTATE =====
            verts = np.array(rotate_y(verts, angle))

            # ===== FACES (for culling only) =====
            faces = [
                [0,1,2],[0,2,3],
                [4,5,6],[4,6,7],
                [0,1,5],[0,5,4],
                [2,3,7],[2,7,6],
                [1,2,6],[1,6,5],
                [0,3,7],[0,7,4]
            ]

            # ===== BACK-FACE CULLING =====
            visible_faces = []
            camera = np.array([0,0,100])

            for f in faces:
                v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]

                normal = np.cross(v1 - v0, v2 - v0)
                if np.linalg.norm(normal) == 0:
                    continue
                normal = normal / np.linalg.norm(normal)

                center = (v0 + v1 + v2) / 3
                view = camera - center
                view = view / np.linalg.norm(view)

                if np.dot(normal, view) > 0:
                    visible_faces.append(f)

            # ===== REAL CUBE EDGES =====
            edge_idx = [
                (0,1),(1,2),(2,3),(3,0),
                (4,5),(5,6),(6,7),(7,4),
                (0,4),(1,5),(2,6),(3,7)
            ]

            # ===== DRAW ONLY TRUE EDGES =====
            for i, j in edge_idx:
                x = [verts[i][0], verts[j][0]]
                y = [verts[i][1], verts[j][1]]
                z = [verts[i][2], verts[j][2]]
                ax.plot(x, y, z, color='black', linewidth=2)

            # ===== VIEW =====
            ax.view_init(elev=20, azim=30)
            ax.set_box_aspect([1,1,1])
            ax.set_xlim(-10,10)
            ax.set_ylim(-10,10)
            ax.set_zlim(-10,10)

            st.pyplot(fig)                                

        else:

            if algorithm == "Painter's Algorithm":

                faces = [
                    [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1)],
                    [(-1,-1, 1),(1,-1, 1),(1,1, 1),(-1,1, 1)],
                    [(-1,-1,-1),(-1,1,-1),(-1,1, 1),(-1,-1, 1)],
                    [(1,-1,-1),(1,-1, 1),(1,1, 1),(1,1,-1)],
                    [(-1,-1,-1),(-1,-1, 1),(1,-1, 1),(1,-1,-1)],
                    [(-1,1,-1),(1,1,-1),(1,1, 1),(-1,1, 1)]
                ]

                # SCALE
                faces = [[(x*size,y*size,z*size) for (x,y,z) in f] for f in faces]

                # ROTATE
                faces = [rotate_y(f, angle) for f in faces]

                # SORT (Painter)
                faces = sorted(
                    faces,
                    key=lambda f: sum(v[2] for v in f)/len(f),
                    reverse=True
                )

                # DRAW (with color + edge 🔥)
                colors = ['red','green','blue','yellow','cyan','magenta']

                for i, face in enumerate(faces):
                    ax.add_collection3d(
                        Poly3DCollection(
                            [face],
                            facecolor=colors[i % 6],
                            edgecolor='black',
                            alpha=0.7
                        )
                    )
                # COMMON AXIS (SAFE)
                ax.set_box_aspect([1,1,1])
                ax.set_xlim(-10,10)
                ax.set_ylim(-10,10)
                ax.set_zlim(-10,10)

                st.pyplot(fig)
                    
elif st.session_state.run_z and algorithm == "Z-Buffer":

    W, H = 500, 350
    img = np.zeros((H, W, 3), dtype=np.float32)
    zbuf = np.full((H, W), np.inf)

    # ===== USER INPUT =====
    dist = st.sidebar.slider("Distance", 200, 1000, 600)
    speed = st.sidebar.slider("Speed", 0.0, 5.0, 1.0)

    if "angle" not in st.session_state:
        st.session_state.angle = 0

    # ===== OBJECTS =====
    def cube(s):
        s /= 2
        return {
            "v": np.array([
                [-s,-s,-s],[s,-s,-s],[s,s,-s],[-s,s,-s],
                [-s,-s,s],[s,-s,s],[s,s,s],[-s,s,s]
            ]),
            "f": [
                [0,1,2],[0,2,3],[4,5,6],[4,6,7],
                [0,1,5],[0,5,4],[2,3,7],[2,7,6],
                [1,2,6],[1,6,5],[0,3,7],[0,7,4]
            ],
            "c": np.array([0.2,0.8,1])
        }

    def pyramid(s):
        s /= 2
        return {
            "v": np.array([
                [-s,-s,-s],[s,-s,-s],[s,-s,s],[-s,-s,s],[0,s,0]
            ]),
            "f": [
                [0,1,2],[0,2,3],
                [0,1,4],[1,2,4],[2,3,4],[3,0,4]
            ],
            "c": np.array([1,0.5,0.2])
        }

    # ===== ROTATION =====
    def rotate(p):
        a = np.radians(st.session_state.angle)
        x,y,z = p

        # Y rotation
        x2 = x*np.cos(a) - z*np.sin(a)
        z2 = x*np.sin(a) + z*np.cos(a)

        # X rotation (true 3D feel 🔥)
        y2 = y*np.cos(a/2) - z2*np.sin(a/2)
        z3 = y*np.sin(a/2) + z2*np.cos(a/2)

        return np.array([x2,y2,z3])

    # ===== PROJECTION (REAL PERSPECTIVE) =====
    def proj(p):
        x,y,z = p

        z += dist
        if z < 1:
            z = 1

        f = 300 / z   # perspective factor

        return np.array([
            x*f + W/2,
            y*f + H/2,
            z
        ])

    # ===== DRAW =====
    def draw_obj(obj, offset):
        tv = np.array([rotate(v) + np.array([offset,0,0]) for v in obj["v"]])

        for f in obj["f"]:
            a,b,c = tv[f[0]], tv[f[1]], tv[f[2]]

            n = normal(a,b,c)
            col = shade(n, obj["c"])

            triangle_zbuffer(
                img, zbuf,
                proj(a), proj(b), proj(c),
                col, W, H
            )

    # ===== DRAW BOTH =====
    draw_obj(cube(size*40), -150)
    draw_obj(pyramid(size*40), 150)

    # ===== UPDATE ROTATION =====
    st.session_state.angle += speed

    st.image(img.astype(np.uint8), caption="Z-Buffer (True 3D)")
    
    time.sleep(0.03)
    st.rerun()  

      
            # st.stop()
st.markdown("---")
st.caption("Computer Graphics Algorithms Simulator • Fully Interactive Version")