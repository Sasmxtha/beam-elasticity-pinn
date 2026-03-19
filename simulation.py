"""
3D Beam Elasticity — Physics-Informed Neural Network (PINN)
===========================================================
Governing Equations (Navier–Cauchy, Eulerian form):
    μ∇²u + (λ+μ)∂/∂x(∇·u)       = 0       [x-direction]
    μ∇²v + (λ+μ)∂/∂y(∇·u)       = 0       [y-direction]
    μ∇²w + (λ+μ)∂/∂z(∇·u) + fz  = 0       [z-direction]

Body force:
    fz = F₀ sin(πx/L)   [sinusoidal transverse load]

Boundary Conditions (fixed ends):
    u = v = w = 0   at x = 0 and x = L

Network: [x,y,z] → 4 × [32, tanh] → [u,v,w]
Validation: ANSYS Mechanical 2025 R1

Authors: Group-17, Amrita Vishwa Vidyapeetham
"""

# ── Dependencies ──────────────────────────────────────────────────────────────
import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# ── Beam Geometry ─────────────────────────────────────────────────────────────
L = 1.0   # Length in x-direction (m)
W = 0.2   # Width  in y-direction (m)
H = 0.2   # Height in z-direction (m)

# ── Material & Load Parameters ────────────────────────────────────────────────
lambda_ = 1.0   # First  Lamé constant (Pa)
mu      = 1.0   # Second Lamé constant — shear modulus (Pa)
F0      = 1.0   # Body-force amplitude (N/m³)

# ── Geometry ──────────────────────────────────────────────────────────────────
geom = dde.geometry.Cuboid([0, 0, 0], [L, W, H])

# ── Governing PDE: Navier–Cauchy (Eulerian) ───────────────────────────────────
def pde(x, u):
    """
    3D Navier–Cauchy equilibrium:
        μ∇²u + (λ+μ)∇(∇·u) + f = 0

    u[:,0] = u-displacement (x)
    u[:,1] = v-displacement (y)
    u[:,2] = w-displacement (z)
    """
    # Second-order derivatives (Laplacians)
    u_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
    u_zz = dde.grad.hessian(u, x, component=0, i=2, j=2)

    v_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)
    v_zz = dde.grad.hessian(u, x, component=1, i=2, j=2)

    w_xx = dde.grad.hessian(u, x, component=2, i=0, j=0)
    w_yy = dde.grad.hessian(u, x, component=2, i=1, j=1)
    w_zz = dde.grad.hessian(u, x, component=2, i=2, j=2)

    # Divergence of displacement field: ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z
    u_x   = dde.grad.jacobian(u, x, i=0, j=0)
    v_y   = dde.grad.jacobian(u, x, i=1, j=1)
    w_z   = dde.grad.jacobian(u, x, i=2, j=2)
    div_u = u_x + v_y + w_z

    # Residuals
    eq_u = mu * (u_xx + u_yy + u_zz) + (lambda_ + mu) * div_u
    eq_v = mu * (v_xx + v_yy + v_zz) + (lambda_ + mu) * div_u
    eq_w = (mu * (w_xx + w_yy + w_zz)
            + (lambda_ + mu) * div_u
            - F0 * tf.sin(np.pi * x[:, 0:1] / L))

    return [eq_u, eq_v, eq_w]

# ── Boundary Conditions: fixed ends at x = 0 and x = L ───────────────────────
def boundary_x(x, on_boundary):
    return on_boundary and (np.isclose(x[0], 0) or np.isclose(x[0], L))

bc_u = dde.DirichletBC(geom, lambda x: 0.0, boundary_x, component=0)
bc_v = dde.DirichletBC(geom, lambda x: 0.0, boundary_x, component=1)
bc_w = dde.DirichletBC(geom, lambda x: 0.0, boundary_x, component=2)

# ── PDE Problem Setup ─────────────────────────────────────────────────────────
data = dde.data.PDE(
    geom,
    pde,
    [bc_u, bc_v, bc_w],
    num_domain=4000,
    num_boundary=200,
    solution=None,
)

# ── Neural Network: [x,y,z] → 4×32 tanh → [u,v,w] ───────────────────────────
net = dde.nn.FNN([3] + [32] * 4 + [3], "tanh", "Glorot normal")

# ── Training ──────────────────────────────────────────────────────────────────
model = dde.Model(data, net)
model.compile("adam", lr=5e-4)
losshistory, train_state = model.train(epochs=20000)

# ── Prediction Grid ───────────────────────────────────────────────────────────
num_points_x = 20
num_points_y = 10
num_points_z = 10

x_vals = np.linspace(0, L, num_points_x)
y_vals = np.linspace(0, W, num_points_y)
z_vals = np.linspace(0, H, num_points_z)
X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
XYZ = np.hstack([X.flatten()[:, None],
                 Y.flatten()[:, None],
                 Z.flatten()[:, None]])

u_pred = model.predict(XYZ)
U      = u_pred[:, 0].reshape(X.shape)
V      = u_pred[:, 1].reshape(X.shape)
W_disp = u_pred[:, 2].reshape(X.shape)

# ── Plot 1: 3D Displacement Vector Field ──────────────────────────────────────
fig    = plt.figure(figsize=(10, 8))
ax     = fig.add_subplot(111, projection='3d')
stride = 2
ax.quiver(
    X[::stride, ::stride, ::stride],
    Y[::stride, ::stride, ::stride],
    Z[::stride, ::stride, ::stride],
    U[::stride, ::stride, ::stride],
    V[::stride, ::stride, ::stride],
    W_disp[::stride, ::stride, ::stride],
    color='royalblue', arrow_length_ratio=0.3
)
ax.set_title("3D Displacement Field (u, v, w) — Navier–Cauchy PINN", fontsize=13)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
ax.grid(True)
plt.tight_layout()
plt.savefig("results/displacement_3d.png", dpi=150)
plt.show()

# ── Plot 2: X-Y Plane Displacement (mid z-slice) ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
mid_z   = num_points_z // 2
ax.quiver(
    X[:, :, mid_z], Y[:, :, mid_z],
    U[:, :, mid_z], V[:, :, mid_z],
    color='royalblue', angles='xy', scale_units='xy', scale=1
)
ax.set_title("X-Y Displacement Field at z = H/2", fontsize=13)
ax.set_xlabel("x (m)", fontsize=12)
ax.set_ylabel("y (m)", fontsize=12)
ax.grid(True)
plt.tight_layout()
plt.savefig("results/displacement_xy.png", dpi=150)
plt.show()

# ── Plot 3: Loss History ──────────────────────────────────────────────────────
dde.utils.plot_loss_history(losshistory)
plt.title("Loss History — 3D Beam PINN")
plt.tight_layout()
plt.savefig("results/3d_loss_history.png", dpi=150)
plt.show()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Max |u| : {np.max(np.abs(U)):.4e} m")
print(f"  Max |v| : {np.max(np.abs(V)):.4e} m")
print(f"  Max |w| : {np.max(np.abs(W_disp)):.4e} m")
print(f"{'='*55}")
