"""
1D Euler-Bernoulli Beam — Physics-Informed Neural Network (PINN)
================================================================
Governing Equation:
    EI * d⁴w/dx⁴ = q(x)

where q(x) is a Gaussian approximation of a central point load:
    q(x) = (P / (σ√2π)) * exp(-(x - L/2)² / (2σ²))

Boundary Conditions (Fixed-Fixed beam):
    w(0) = 0,  w(L) = 0   (zero displacement)
    w'(0) = 0, w'(L) = 0  (zero slope)

Material: Aluminum
Validation: ANSYS Mechanical 2025 R1

Authors: Group-17, Amrita Vishwa Vidyapeetham
"""

# ── Dependencies ──────────────────────────────────────────────────────────────
import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# ── Beam & Material Parameters ────────────────────────────────────────────────
L     = 1.0       # Beam length (m)
b     = 0.02      # Width (m)
h     = 0.005     # Thickness (m)
E     = 70e9      # Young's modulus — Aluminum (Pa)
I     = b * h**3 / 12   # Second moment of area (m⁴)
EI    = E * I     # Flexural rigidity (N·m²)

P     = 100.0     # Applied central point load (N)
sigma = 0.01      # Gaussian width for Dirac delta approximation (m)

print(f"Beam: L={L} m | b={b} m | h={h} m")
print(f"Material: E={E:.2e} Pa | I={I:.4e} m⁴ | EI={EI:.4e} N·m²")

# ── Gaussian Load: Approximation of central point load ────────────────────────
def q_load(x):
    """
    Distributed load approximating a Dirac delta at x = L/2
    q(x) = (P / σ√(2π)) * exp(-(x - L/2)² / 2σ²)
    """
    return (P / (tf.sqrt(2.0 * np.pi) * sigma)) * \
           tf.exp(-0.5 * ((x[:, 0:1] - L / 2.0) / sigma) ** 2)

# ── Governing PDE: EI * d⁴w/dx⁴ = q(x) ──────────────────────────────────────
def pde(x, w):
    """
    4th-order Euler-Bernoulli beam equation.
    Computed via nested Hessian (2nd derivative of 2nd derivative).
    """
    d2w_dx2 = dde.grad.hessian(w, x, i=0, j=0)
    d4w_dx4 = dde.grad.hessian(d2w_dx2, x, i=0, j=0)
    return EI * d4w_dx4 - q_load(x)

# ── Geometry: 1D interval [0, L] ──────────────────────────────────────────────
geom = dde.geometry.Interval(0, L)

# ── Boundary Conditions ───────────────────────────────────────────────────────
# Dirichlet: zero displacement at both ends
bc_w0 = dde.DirichletBC(
    geom,
    lambda x: np.zeros((x.shape[0], 1)),
    lambda x, on_boundary: on_boundary and np.isclose(x[0], 0)
)
bc_wL = dde.DirichletBC(
    geom,
    lambda x: np.zeros((x.shape[0], 1)),
    lambda x, on_boundary: on_boundary and np.isclose(x[0], L)
)

# Neumann: zero slope (fixed end — zero rotation) at both ends
def zero_slope(x, w, _):
    return dde.grad.jacobian(w, x, i=0, j=0)

bc_dw0 = dde.OperatorBC(
    geom,
    zero_slope,
    lambda x, on_boundary: on_boundary and np.isclose(x[0], 0)
)
bc_dwL = dde.OperatorBC(
    geom,
    zero_slope,
    lambda x, on_boundary: on_boundary and np.isclose(x[0], L)
)

# ── PDE Problem Setup ─────────────────────────────────────────────────────────
data = dde.data.PDE(
    geom,
    pde,
    [bc_w0, bc_wL, bc_dw0, bc_dwL],
    num_domain=400,
    num_boundary=20,
    solution=None,
)

# ── Neural Network: 1 input (x) → 3 hidden × 64 → 1 output (w) ───────────────
net = dde.nn.FNN([1] + [64] * 3 + [1], "tanh", "Glorot uniform")

# ── Training: Adam first, then L-BFGS-B ──────────────────────────────────────
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(epochs=8000)

model.compile("L-BFGS-B")
losshistory, train_state = model.train()

# ── Prediction ────────────────────────────────────────────────────────────────
x_pred = np.linspace(0, L, 500)[:, None]
w_pred = model.predict(x_pred)

# ── Plot 1: Deflection Profile ────────────────────────────────────────────────
plt.figure(figsize=(9, 5))
plt.plot(x_pred, w_pred * 1000, 'b-', linewidth=2, label="PINN Prediction")
plt.axvline(x=L/2, color='r', linestyle='--', alpha=0.6, label="Load position (x = L/2)")
plt.xlabel("x (m)", fontsize=12)
plt.ylabel("Deflection w(x) × 10³ (mm)", fontsize=12)
plt.title("1D Beam Deflection — Euler-Bernoulli PINN\n(Fixed-Fixed, Central Point Load)", fontsize=13)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/1d_deflection.png", dpi=150)
plt.show()

# ── Plot 2: PINN vs ANSYS Comparison ─────────────────────────────────────────
# ANSYS reference: max deflection at x=L/2 is -0.034985 m
# Scaled for comparison (ANSYS used P=100N, same geometry)
ansys_max = -0.034985

plt.figure(figsize=(9, 5))
plt.plot(x_pred, w_pred, 'b-', linewidth=2, label="PINN Prediction")
plt.axhline(y=ansys_max, color='r', linestyle='--', linewidth=1.5,
            label=f"ANSYS Max Deflection = {ansys_max:.4f} m")
plt.axvline(x=L/2, color='gray', linestyle=':', alpha=0.6, label="Center (x = L/2)")
plt.xlabel("x (m)", fontsize=12)
plt.ylabel("Deflection w(x) (m)", fontsize=12)
plt.title("PINN vs ANSYS — 1D Beam Deflection", fontsize=13)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/1d_pinn_vs_ansys.png", dpi=150)
plt.show()

# ── Plot 3: Loss History ──────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.semilogy(losshistory.steps, np.sum(losshistory.loss_train, axis=1),
             'b-', linewidth=1.5, label="Training Loss")
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss (log scale)", fontsize=12)
plt.title("Training Loss History — 1D Beam PINN", fontsize=13)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/1d_loss_history.png", dpi=150)
plt.show()

# ── Summary ───────────────────────────────────────────────────────────────────
w_max = float(np.min(w_pred))
x_max = float(x_pred[np.argmin(w_pred)])
print(f"\n{'='*50}")
print(f"  PINN Max Deflection : {w_max:.6f} m at x = {x_max:.4f} m")
print(f"  ANSYS Max Deflection: {ansys_max:.6f} m at x = {L/2:.4f} m")
print(f"  Absolute Error      : {abs(w_max - ansys_max):.6f} m")
print(f"{'='*50}")
