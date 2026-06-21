import numpy as np
from typing import Tuple, List, Optional
from .geometry import Geometry1D, Geometry2D
from .solver_1g import thomas_solve


def solve_1d_diffusion_2g(
    geom: Geometry1D,
    fission_source_1: np.ndarray,
    fission_source_2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    x_nodes, x_centers, material_names = geom.build_mesh()
    materials = geom.get_materials_2g()
    n = len(x_centers)
    dx = geom.dx
    (d_left_1, d_left_2), (d_right_1, d_right_2) = geom.get_extrapolation_distances_2g()

    D1 = np.array([m.D1 for m in materials])
    Sigma_a1 = np.array([m.Sigma_a1 for m in materials])
    Sigma_s12 = np.array([m.Sigma_s12 for m in materials])
    D2 = np.array([m.D2 for m in materials])
    Sigma_a2 = np.array([m.Sigma_a2 for m in materials])

    D1_interface = np.zeros(n + 1)
    D2_interface = np.zeros(n + 1)
    for i in range(1, n):
        D1_interface[i] = 2.0 * D1[i - 1] * D1[i] / (D1[i - 1] + D1[i])
        D2_interface[i] = 2.0 * D2[i - 1] * D2[i] / (D2[i - 1] + D2[i])
    D1_interface[0] = D1[0]
    D1_interface[n] = D1[n - 1]
    D2_interface[0] = D2[0]
    D2_interface[n] = D2[n - 1]

    a1 = np.zeros(n)
    b1 = np.zeros(n)
    c1 = np.zeros(n)
    rhs1 = fission_source_1.copy()

    for i in range(n):
        b1[i] = Sigma_a1[i] + Sigma_s12[i] + (D1_interface[i] + D1_interface[i + 1]) / (dx ** 2)
        if i > 0:
            a1[i] = -D1_interface[i] / (dx ** 2)
        if i < n - 1:
            c1[i] = -D1_interface[i + 1] / (dx ** 2)

    if geom.bc.left == "zero_flux":
        b1[0] += D1_interface[0] / (dx ** 2)
    elif geom.bc.left == "vacuum":
        coeff = 2.0 * D1_interface[0] / (dx * (dx + 2.0 * d_left_1))
        b1[0] += coeff

    if geom.bc.right == "zero_flux":
        b1[n - 1] += D1_interface[n] / (dx ** 2)
    elif geom.bc.right == "vacuum":
        coeff = 2.0 * D1_interface[n] / (dx * (dx + 2.0 * d_right_1))
        b1[n - 1] += coeff

    phi1 = thomas_solve(a1, b1, c1, rhs1)
    phi1 = np.maximum(phi1, 1e-20)

    a2 = np.zeros(n)
    b2 = np.zeros(n)
    c2 = np.zeros(n)
    rhs2 = fission_source_2 + Sigma_s12 * phi1

    for i in range(n):
        b2[i] = Sigma_a2[i] + (D2_interface[i] + D2_interface[i + 1]) / (dx ** 2)
        if i > 0:
            a2[i] = -D2_interface[i] / (dx ** 2)
        if i < n - 1:
            c2[i] = -D2_interface[i + 1] / (dx ** 2)

    if geom.bc.left == "zero_flux":
        b2[0] += D2_interface[0] / (dx ** 2)
    elif geom.bc.left == "vacuum":
        coeff = 2.0 * D2_interface[0] / (dx * (dx + 2.0 * d_left_2))
        b2[0] += coeff

    if geom.bc.right == "zero_flux":
        b2[n - 1] += D2_interface[n] / (dx ** 2)
    elif geom.bc.right == "vacuum":
        coeff = 2.0 * D2_interface[n] / (dx * (dx + 2.0 * d_right_2))
        b2[n - 1] += coeff

    phi2 = thomas_solve(a2, b2, c2, rhs2)
    phi2 = np.maximum(phi2, 1e-20)

    return phi1, phi2


def solve_2d_diffusion_2g_sor(
    geom: Geometry2D,
    fission_source_1: np.ndarray,
    fission_source_2: np.ndarray,
    omega: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 10000,
    initial_guess1: Optional[np.ndarray] = None,
    initial_guess2: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int, List[float]]:
    x_nodes, y_nodes, x_centers, y_centers, material_names = geom.build_mesh()
    materials = geom.get_materials_2g()
    ny, nx = len(y_centers), len(x_centers)
    dx, dy = geom.dx, geom.dy
    (d_left_1, d_right_1, d_bottom_1, d_top_1), \
    (d_left_2, d_right_2, d_bottom_2, d_top_2) = geom.get_extrapolation_distances_2g()

    D1 = np.array([[m.D1 for m in row] for row in materials])
    Sigma_a1 = np.array([[m.Sigma_a1 for m in row] for row in materials])
    Sigma_s12 = np.array([[m.Sigma_s12 for m in row] for row in materials])
    D2 = np.array([[m.D2 for m in row] for row in materials])
    Sigma_a2 = np.array([[m.Sigma_a2 for m in row] for row in materials])

    if initial_guess1 is None:
        phi1 = np.ones((ny, nx))
    else:
        phi1 = initial_guess1.copy()
    if initial_guess2 is None:
        phi2 = np.ones((ny, nx))
    else:
        phi2 = initial_guess2.copy()

    phi1 = np.maximum(phi1, 1e-20)
    phi2 = np.maximum(phi2, 1e-20)
    S1 = fission_source_1.reshape((ny, nx))
    S2 = fission_source_2.reshape((ny, nx))

    D1_x = np.zeros((ny, nx + 1))
    D2_x = np.zeros((ny, nx + 1))
    for j in range(ny):
        for i in range(1, nx):
            D1_x[j, i] = 2.0 * D1[j, i - 1] * D1[j, i] / (D1[j, i - 1] + D1[j, i])
            D2_x[j, i] = 2.0 * D2[j, i - 1] * D2[j, i] / (D2[j, i - 1] + D2[j, i])
        D1_x[j, 0] = D1[j, 0]
        D1_x[j, nx] = D1[j, nx - 1]
        D2_x[j, 0] = D2[j, 0]
        D2_x[j, nx] = D2[j, nx - 1]

    D1_y = np.zeros((ny + 1, nx))
    D2_y = np.zeros((ny + 1, nx))
    for j in range(1, ny):
        for i in range(nx):
            D1_y[j, i] = 2.0 * D1[j - 1, i] * D1[j, i] / (D1[j - 1, i] + D1[j, i])
            D2_y[j, i] = 2.0 * D2[j - 1, i] * D2[j, i] / (D2[j - 1, i] + D2[j, i])
    for i in range(nx):
        D1_y[0, i] = D1[0, i]
        D1_y[ny, i] = D1[ny - 1, i]
        D2_y[0, i] = D2[0, i]
        D2_y[ny, i] = D2[ny - 1, i]

    def gs_update(phi, D, Sigma_a, S, extra_source, d_l, d_r, d_b, d_t, D_x, D_y):
        for j in range(ny):
            for i in range(nx):
                diag = Sigma_a[j, i]
                val = S[j, i] + extra_source[j, i]

                if i > 0:
                    coeff = D_x[j, i] / (dx ** 2)
                    val += coeff * phi[j, i - 1]
                    diag += coeff
                else:
                    if geom.bc.left == "zero_flux":
                        coeff = D_x[j, 0] / (dx ** 2)
                        diag += coeff
                    elif geom.bc.left == "vacuum":
                        coeff = 2.0 * D_x[j, 0] / (dx * (dx + 2.0 * d_l))
                        diag += coeff
                    elif geom.bc.left == "reflective":
                        coeff = D_x[j, 0] / (dx ** 2)
                        mirror_idx = i + 1 if i + 1 < nx else i
                        val += coeff * phi[j, mirror_idx]
                        diag += coeff

                if i < nx - 1:
                    coeff = D_x[j, i + 1] / (dx ** 2)
                    val += coeff * phi[j, i + 1]
                    diag += coeff
                else:
                    if geom.bc.right == "zero_flux":
                        coeff = D_x[j, nx] / (dx ** 2)
                        diag += coeff
                    elif geom.bc.right == "vacuum":
                        coeff = 2.0 * D_x[j, nx] / (dx * (dx + 2.0 * d_r))
                        diag += coeff
                    elif geom.bc.right == "reflective":
                        coeff = D_x[j, nx] / (dx ** 2)
                        mirror_idx = i - 1 if i - 1 >= 0 else i
                        val += coeff * phi[j, mirror_idx]
                        diag += coeff

                if j > 0:
                    coeff = D_y[j, i] / (dy ** 2)
                    val += coeff * phi[j - 1, i]
                    diag += coeff
                else:
                    if geom.bc.bottom == "zero_flux":
                        coeff = D_y[0, i] / (dy ** 2)
                        diag += coeff
                    elif geom.bc.bottom == "vacuum":
                        coeff = 2.0 * D_y[0, i] / (dy * (dy + 2.0 * d_b))
                        diag += coeff
                    elif geom.bc.bottom == "reflective":
                        coeff = D_y[0, i] / (dy ** 2)
                        mirror_idx = j + 1 if j + 1 < ny else j
                        val += coeff * phi[mirror_idx, i]
                        diag += coeff

                if j < ny - 1:
                    coeff = D_y[j + 1, i] / (dy ** 2)
                    val += coeff * phi[j + 1, i]
                    diag += coeff
                else:
                    if geom.bc.top == "zero_flux":
                        coeff = D_y[ny, i] / (dy ** 2)
                        diag += coeff
                    elif geom.bc.top == "vacuum":
                        coeff = 2.0 * D_y[ny, i] / (dy * (dy + 2.0 * d_t))
                        diag += coeff
                    elif geom.bc.top == "reflective":
                        coeff = D_y[ny, i] / (dy ** 2)
                        mirror_idx = j - 1 if j - 1 >= 0 else j
                        val += coeff * phi[mirror_idx, i]
                        diag += coeff

                updated = val / diag
                phi[j, i] = phi[j, i] + omega * (updated - phi[j, i])
                phi[j, i] = max(phi[j, i], 1e-20)

    residuals = []
    for iteration in range(max_iter):
        phi1_old = phi1.copy()
        phi2_old = phi2.copy()

        extra1 = np.zeros((ny, nx))
        gs_update(phi1, D1, Sigma_a1 + Sigma_s12, S1, extra1,
                  d_left_1, d_right_1, d_bottom_1, d_top_1, D1_x, D1_y)

        extra2 = Sigma_s12 * phi1
        gs_update(phi2, D2, Sigma_a2, S2, extra2,
                  d_left_2, d_right_2, d_bottom_2, d_top_2, D2_x, D2_y)

        change1 = np.max(np.abs(phi1 - phi1_old) / np.maximum(np.abs(phi1_old), 1e-20))
        change2 = np.max(np.abs(phi2 - phi2_old) / np.maximum(np.abs(phi2_old), 1e-20))
        change = max(change1, change2)
        residuals.append(change)
        if change < tol:
            return phi1, phi2, iteration + 1, residuals

    return phi1, phi2, max_iter, residuals
