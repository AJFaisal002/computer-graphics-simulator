import streamlit as st
import matplotlib.pyplot as plt
from algorithms import (
    dda_line,
    bresenham_line,
    midpoint_circle
)

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Computer Graphics Algorithms Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# HEADER
# ==========================================================
st.title("🖥️ Computer Graphics Algorithms Simulator")
st.caption("DDA Line • Bresenham Line • Midpoint Circle")

st.markdown("---")

# ==========================================================
# SIDEBAR CONTROLS
# ==========================================================
st.sidebar.header("⚙️ Algorithm Controls")

algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["DDA Line", "Bresenham Line", "Midpoint Circle"]
)

# ==========================================================
# INPUT PARAMETERS
# ==========================================================
if algorithm in ["DDA Line", "Bresenham Line"]:
    st.sidebar.subheader("Line Parameters")

    x0 = st.sidebar.number_input("x₀ (Start X)", value=2)
    y0 = st.sidebar.number_input("y₀ (Start Y)", value=3)
    x1 = st.sidebar.number_input("x₁ (End X)", value=12)
    y1 = st.sidebar.number_input("y₁ (End Y)", value=8)

else:
    st.sidebar.subheader("Circle Parameters")

    xc = st.sidebar.number_input("Center X", value=0)
    yc = st.sidebar.number_input("Center Y", value=0)
    r  = st.sidebar.number_input("Radius", value=10, min_value=1)

draw_button = st.sidebar.button("▶ Draw", use_container_width=True)

# ==========================================================
# MAIN PLOTTING AREA
# ==========================================================
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.grid(True, linestyle="--", alpha=0.6)
ax.set_title(algorithm)

# ==========================================================
# DRAWING LOGIC
# ==========================================================
if draw_button:
    if algorithm == "DDA Line":
        points = dda_line(x0, y0, x1, y1)
    elif algorithm == "Bresenham Line":
        points = bresenham_line(x0, y0, x1, y1)
    else:
        points = midpoint_circle(xc, yc, r)

    if points:
        x_vals, y_vals = zip(*points)
        ax.scatter(x_vals, y_vals, s=25)
        st.pyplot(fig)
    else:
        st.warning("No points generated. Please check input values.")

# ==========================================================
# ALGORITHM DESCRIPTION (PROFESSIONAL TOUCH)
# ==========================================================
with st.expander("📘 Algorithm Description"):
    if algorithm == "DDA Line":
        st.write(
            """
            **Digital Differential Analyzer (DDA)** is a line drawing algorithm that
            incrementally plots points between two endpoints using the slope of the line.
            It uses floating-point arithmetic and is simple to implement.
            """
        )

    elif algorithm == "Bresenham Line":
        st.write(
            """
            **Bresenham’s Line Algorithm** is an efficient rasterization technique
            that uses only integer arithmetic and a decision parameter to determine
            the next pixel position.
            """
        )

    else:
        st.write(
            """
            **Midpoint Circle Algorithm** draws a circle using symmetry and a
            decision parameter. It efficiently determines pixel positions without
            trigonometric calculations.
            """
        )

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.caption("Computer Graphics Algorithms Simulator • Built with Streamlit")
