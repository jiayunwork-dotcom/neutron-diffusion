import numpy as np
from typing import Tuple, List, Optional
from .geometry import Geometry1D, Geometry2D
from .materials import Material1G


def thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = len(d)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    x = np.zeros(n)
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n):
        m = 1.0 / (b[i] - a[i] * c_prime[i - 1])
        c_prime[i] = c[i] * m
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) * m
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def solve_1d_diffusion_1g(
    geom: Geometry1D,
    source: np.ndarray,
) -> np.ndarray:
    x_nodes, x_centers, material_names = geom.build_mesh()
    materials = geom.get_materials_1g()
    n = len(x_centers)
    dx = geom.dx
    d_left, d_right = geom.get_extrapolation_distances_1g()

    D = np.array([m.D for m in materials])
    Sigma_a = np.array([m.Sigma_a for m in materials])

    D_interface = np.zeros(n + 1)
    for i in range(1, n):
        D_interface[i] = 2.0 * D[i - 1] * D[i] / (D[i - 1] + D[i])
    D_interface[0] = D[0]
    D_interface[n] = D[n - 1]

    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    rhs = source.copy()

    for i in range(n):
        b[i] = Sigma_a[i] + (D_interface[i] + D_interface[i + 1]) / (dx ** 2)
        if i > 0:
            a[i] = -D_interface[i] / (dx ** 2)
        if i < n - 1:
            c[i] = -D_interface[i + 1] / (dx ** 2)

    if geom.bc.left == "zero_flux":
        b[0] += D_interface[0] / (dx ** 2)
    elif geom.bc.left == "reflective":
        pass
    elif geom.bc.left == "vacuum":
        d_eff = d_left
        coeff = 2.0 * D_interface[0] / (dx * (dx + 2.0 * d_eff))
        b[0] += coeff

    if geom.bc.right == "zero_flux":
        b[n - 1] += D_interface[n] / (dx ** 2)
    elif geom.bc.right == "reflective":
        pass
    elif geom.bc.right == "vacuum":
        d_eff = d_right
        coeff = 2.0 * D_interface[n] / (dx * (dx + 2.0 * d_eff))
        b[n - 1] += coeff

    phi = thomas_solve(a, b, c, rhs)
    phi = np.maximum(phi, 1e-20)
    return phi


def solve_2d_diffusion_1g_sor(
    geom: Geometry2D,
    source: np.ndarray,
    omega: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 10000,
    initial_guess: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, List[float]]:
    x_nodes, y_nodes, x_centers, y_centers, material_names = geom.build_mesh()
    materials = geom.get_materials_1g()
    ny, nx = len(y_centers), len(x_centers)
    dx, dy = geom.dx, geom.dy
    d_left, d_right, d_bottom, d_top = geom.get_extrapolation_distances_1g()

    D = np.array([[m.D for m in row] for row in materials])
    Sigma_a = np.array([[m.Sigma_a for m in row] for row in materials])

    if initial_guess is None:
        phi = np.ones((ny, nx))
    else:
        phi = initial_guess.copy()

    phi = np.maximum(phi, 1e-20)
    S = source.reshape((ny, nx))

    D_x = np.zeros((ny, nx + 1))
    for j in range(ny):
        for i in range(1, nx):
            D_x[j, i] = 2.0 * D[j, i - 1] * D[j, i] / (D[j, i - 1] + D[j, i])
        D_x[j, 0] = D[j, 0]
        D_x[j, nx] = D[j, nx - 1]

    D_y = np.zeros((ny + 1, nx))
    for j in range(1, ny):
        for i in range(nx):
            D_y[j, i] = 2.0 * D[j - 1, i] * D[j, i] / (D[j - 1, i] + D[j, i])
    for i in range(nx):
        D_y[0, i] = D[0, i]
        D_y[ny, i] = D[ny - 1, i]

    residuals = []
    for iteration in range(max_iter):
        phi_old = phi.copy()
        max_change = 0.0

        for j in range(ny):
            for i in range(nx):
                diag = Sigma_a[j, i]
                phi_new = S[j, i]

                if i > 0:
                    coeff = D_x[j, i] / (dx ** 2)
                    phi_new += coeff * phi[j, i - 1]
                    diag += coeff
                else:
                    if geom.bc.left == "zero_flux":
                        coeff = D_x[j, 0] / (dx ** 2)
                        diag += coeff
                    elif geom.bc.left == "vacuum":
                        coeff = 2.0 * D_x[j, 0] / (dx * (dx + 2.0 * d_left))
                        diag += coeff
                    elif geom.bc.left == "reflective":
                        coeff = D_x[j, 0] / (dx ** 2)
                        mirror_idx = i + 1 if i + 1 < nx else i
                        phi_new += coeff * phi[j, mirror_idx]
                        diag += coeff

                if i < nx - 1:
                    coeff = D_x[j, i + 1] / (dx ** 2)
                    phi_new += coeff * phi[j, i + 1]
                    diag += coeff
                else:
                    if geom.bc.right == "zero_flux":
                        coeff = D_x[j, nx] / (dx ** 2)
                        diag += coeff
                    elif geom.bc.right == "vacuum":
                        coeff = 2.0 * D_x[j, nx] / (dx * (dx + 2.0 * d_right))
                        diag += coeff
                    elif geom.bc.right == "reflective":
                        coeff = D_x[j, nx] / (dx ** 2)
                        mirror_idx = i - 1 if i - 1 >= 0 else i
                        phi_new += coeff * phi[j, mirror_idx]
                        diag += coeff

                if j > 0:
                    coeff = D_y[j, i] / (dy ** 2)
                    phi_new += coeff * phi[j - 1, i]
                    diag += coeff
                else:
                    if geom.bc.bottom == "zero_flux":
                        coeff = D_y[0, i] / (dy ** 2)
                        diag += coeff
                    elif geom.bc.bottom == "vacuum":
                        coeff = 2.0 * D_y[0, i] / (dy * (dy + 2.0 * d_bottom))
                        diag += coeff
                    elif geom.bc.bottom == "reflective":
                        coeff = D_y[0, i] / (dy ** 2)
                        mirror_idx = j + 1 if j + 1 < ny else j
                        phi_new += coeff * phi[mirror_idx, i]
                        diag += coeff

                if j < ny - 1:
                    coeff = D_y[j + 1, i] / (dy ** 2)
                    phi_new += coeff * phi[j + 1, i]
                    diag += coeff
                else:
                    if geom.bc.top == "zero_flux":
                        coeff = D_y[ny, i] / (dy ** 2)
                        diag += coeff
                    elif geom.bc.top == "vacuum":
                        coeff = 2.0 * D_y[ny, i] / (dy * (dy + 2.0 * d_top))
                        diag += coeff
                    elif geom.bc.top == "reflective":
                        coeff = D_y[ny, i] / (dy ** 2)
                        mirror_idx = j - 1 if j - 1 >= 0 else j
                        phi_new += coeff * phi[mirror_idx, i]
                        diag += coeff

                phi_new /= diag
                phi[j, i] = phi[j, i] + omega * (phi_new - phi[j, i])
                phi[j, i] = max(phi[j, i], 1e-20)

        change = np.max(np.abs(phi - phi_old) / np.maximum(np.abs(phi_old), 1e-20))
        residuals.append(change)
        if change < tol:
            return phi, iteration + 1, residuals

    return phi, max_iter, residuals
