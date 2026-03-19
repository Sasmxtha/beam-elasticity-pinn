# ANSYS vs PINN — Comparison Notes

## Tools Used
- **ANSYS Mechanical 2025 R1** (Student Edition)
- **DeepXDE** with TensorFlow backend

---

## Case 1 — 1D Aluminum Beam

### ANSYS Setup
- Solver: Static Structural
- Type: Directional Deformation (Y Axis)
- Unit: meters
- Date: 22-04-2025

### ANSYS Results
| Metric | Value |
|--------|-------|
| Max Deformation | 1.0417e-6 m (tip) |
| Min Deformation | −0.034985 m (center) |

### Comparison
| Metric | ANSYS | PINN |
|--------|-------|------|
| Max deflection at center | −0.034985 m | ~−0.035 m |
| Profile shape | Symmetric bell curve | ✅ Matches |
| Boundary conditions satisfied | ✅ | ✅ |

---

## Case 2 — 2D Aluminum Plate

### ANSYS Setup
- Solver: Static Structural
- Type: Directional Deformation (Z Axis)
- Unit: meters
- Date: 22-04-2025

### ANSYS Results
| Metric | Value |
|--------|-------|
| Max Deformation | 0.0026818 m |
| Min Deformation | −3.154e-5 m |

### Comparison
| Metric | ANSYS | PINN |
|--------|-------|------|
| Max deflection (center) | 0.0026818 m | ~0.0026 m |
| Validation MSE | — | 4.09e-08 |
| Validation R² Score | — | 0.94691 |
| Profile shape (radially symmetric) | ✅ | ✅ |

---

## Case 3 — 3D Elastic Beam

- No direct ANSYS comparison for 3D case (used as extension study)
- Results show physically consistent displacement field (u, v, w)
- Boundary conditions (zero displacement at x=0 and x=L) are satisfied

---

## Images in this folder

| File | Description |
|------|-------------|
| `1d_ansys_deformation.png` | 1D beam Y-axis directional deformation from ANSYS |
| `2d_ansys_deformation.png` | 2D plate Z-axis directional deformation from ANSYS |
