import plotly.graph_objects as go
import numpy as np

def plot_bloch_sphere(bloch_vector, qubit_index):
    x, y, z = bloch_vector["x"], bloch_vector["y"], bloch_vector["z"]

    fig = go.Figure()

    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    fig.add_surface(x=xs, y=ys, z=zs, opacity=0.1, colorscale="Blues")

    fig.add_trace(go.Scatter3d(
        x=[0, x], y=[0, y], z=[0, z],
        mode="lines+markers",
        line=dict(color="red", width=5),
        marker=dict(size=4, color="red")
    ))

    fig.update_layout(
        title=f"Qubit {qubit_index} Bloch Vector",
        scene=dict(
            xaxis=dict(range=[-1,1]),
            yaxis=dict(range=[-1,1]),
            zaxis=dict(range=[-1,1])
        )
    )
    return fig
