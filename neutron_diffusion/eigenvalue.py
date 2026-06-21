import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from .geometry import Geometry1D, Geometry2D
from .solver_1g import solve_1d_diffusion_1g, solve_2d_diffusion_1g_sor
from .solver_2g import solve_1d_diffusion_2g, solve_2d_diffusion_2g_sor


@dataclass
class EigenvalueResult:
    keff: float
    keff_history: List[float] = field(default_factory=list)
    phi: Optional[np.ndarray] = None
    phi1: Optional[np.ndarray] = None
    phi2: Optional[np.ndarray] = None
    converged: bool = False
    n_iterations: int = 0
    inner_iterations: List[int] = field(default_factory=list)
    flux_changes: List[float] = field(default_factory=list)


def criticality_1d_1g(
    geom: Geometry1D,
    k_tol: float = 1e-5,
    flux_tol: float = 1e-4,
    max_iter: int = 500,
) -> EigenvalueResult:
    _, x_centers, _ = geom.build_mesh()
    materials = geom.get_materials_1g()
    n = len(x_centers)
    nuSigma_f = np.array([m.nuSigma_f for m in materials])

    phi = np.ones(n)
    k_old = 1.0
    result = EigenvalueResult(keff=k_old)
    result.keff_history.append(k_old)

    for iteration in range(max_iter):
        S = nuSigma_f * phi / k_old
        phi_new = solve_1d_diffusion_1g(geom, S)

        numerator = np.sum(nuSigma_f * phi_new)
        denominator = np.sum(nuSigma_f * phi)
        k_new = k_old * numerator / denominator if denominator > 0 else k_old

        phi_max = np.max(phi_new)
        if phi_max > 1e-30:
            phi_new = phi_new / phi_max

        k_error = abs(k_new - k_old) / k_old if k_old > 0 else 1.0
        flux_error = np.max(np.abs(phi_new - phi) / np.maximum(phi, 1e-20))

        result.keff_history.append(k_new)
        result.flux_changes.append(flux_error)

        if k_error < k_tol and flux_error < flux_tol:
            phi = phi_new
            k_old = k_new
            result.converged = True
            result.n_iterations = iteration + 1
            break

        phi = phi_new
        k_old = k_new
        result.n_iterations = iteration + 1

    result.keff = k_old
    result.phi = phi
    return result


def criticality_2d_1g(
    geom: Geometry2D,
    omega: float = 1.0,
    inner_tol: float = 1e-6,
    inner_max_iter: int = 2000,
    k_tol: float = 1e-5,
    flux_tol: float = 1e-4,
    max_iter: int = 500,
) -> EigenvalueResult:
    _, _, x_centers, y_centers, _ = geom.build_mesh()
    materials = geom.get_materials_1g()
    ny, nx = len(y_centers), len(x_centers)
    nuSigma_f = np.array([[m.nuSigma_f for m in row] for row in materials])

    phi = np.ones((ny, nx))
    k_old = 1.0
    result = EigenvalueResult(keff=k_old)
    result.keff_history.append(k_old)

    for iteration in range(max_iter):
        S = (nuSigma_f * phi / k_old).flatten()
        phi_new, inner_it, _ = solve_2d_diffusion_1g_sor(
            geom, S, omega=omega, tol=inner_tol,
            max_iter=inner_max_iter, initial_guess=phi.copy()
        )
        result.inner_iterations.append(inner_it)

        if not np.all(np.isfinite(phi_new)):
            phi_new = np.ones((ny, nx))

        numerator = np.sum(nuSigma_f * phi_new)
        denominator = np.sum(nuSigma_f * phi)
        k_new = k_old * numerator / denominator if denominator > 0 else k_old
        if not np.isfinite(k_new):
            k_new = k_old

        phi_max = np.max(phi_new)
        if phi_max > 1e-30:
            phi_new = phi_new / phi_max
        else:
            phi_new = np.ones((ny, nx))

        k_error = abs(k_new - k_old) / k_old if k_old > 0 else 1.0
        flux_error = np.max(np.abs(phi_new - phi) / np.maximum(phi, 1e-20))
        if not np.isfinite(k_error):
            k_error = 1.0
        if not np.isfinite(flux_error):
            flux_error = 1.0

        result.keff_history.append(k_new)
        result.flux_changes.append(flux_error)

        if k_error < k_tol and flux_error < flux_tol:
            phi = phi_new
            k_old = k_new
            result.converged = True
            result.n_iterations = iteration + 1
            break

        phi = phi_new
        k_old = k_new
        result.n_iterations = iteration + 1

    result.keff = k_old
    result.phi = phi
    return result


def criticality_1d_2g(
    geom: Geometry1D,
    k_tol: float = 1e-5,
    flux_tol: float = 1e-4,
    max_iter: int = 500,
) -> EigenvalueResult:
    _, x_centers, _ = geom.build_mesh()
    materials = geom.get_materials_2g()
    n = len(x_centers)
    nuSigma_f1 = np.array([m.nuSigma_f1 for m in materials])
    nuSigma_f2 = np.array([m.nuSigma_f2 for m in materials])

    phi1 = np.ones(n)
    phi2 = np.ones(n)
    k_old = 1.0
    result = EigenvalueResult(keff=k_old)
    result.keff_history.append(k_old)

    for iteration in range(max_iter):
        S1 = (nuSigma_f1 * phi1 + nuSigma_f2 * phi2) / k_old
        S2 = np.zeros(n)
        phi1_new, phi2_new = solve_1d_diffusion_2g(geom, S1, S2)

        numerator = np.sum(nuSigma_f1 * phi1_new + nuSigma_f2 * phi2_new)
        denominator = np.sum(nuSigma_f1 * phi1 + nuSigma_f2 * phi2)
        k_new = k_old * numerator / denominator if denominator > 0 else k_old

        phi_max = max(np.max(phi1_new), np.max(phi2_new))
        if phi_max > 1e-30:
            phi1_new = phi1_new / phi_max
            phi2_new = phi2_new / phi_max

        k_error = abs(k_new - k_old) / k_old if k_old > 0 else 1.0
        flux_error1 = np.max(np.abs(phi1_new - phi1) / np.maximum(phi1, 1e-20))
        flux_error2 = np.max(np.abs(phi2_new - phi2) / np.maximum(phi2, 1e-20))
        flux_error = max(flux_error1, flux_error2)

        result.keff_history.append(k_new)
        result.flux_changes.append(flux_error)

        if k_error < k_tol and flux_error < flux_tol:
            phi1, phi2 = phi1_new, phi2_new
            k_old = k_new
            result.converged = True
            result.n_iterations = iteration + 1
            break

        phi1, phi2 = phi1_new, phi2_new
        k_old = k_new
        result.n_iterations = iteration + 1

    result.keff = k_old
    result.phi1 = phi1
    result.phi2 = phi2
    return result


def criticality_2d_2g(
    geom: Geometry2D,
    omega: float = 1.0,
    inner_tol: float = 1e-6,
    inner_max_iter: int = 2000,
    k_tol: float = 1e-5,
    flux_tol: float = 1e-4,
    max_iter: int = 500,
) -> EigenvalueResult:
    _, _, x_centers, y_centers, _ = geom.build_mesh()
    materials = geom.get_materials_2g()
    ny, nx = len(y_centers), len(x_centers)
    nuSigma_f1 = np.array([[m.nuSigma_f1 for m in row] for row in materials])
    nuSigma_f2 = np.array([[m.nuSigma_f2 for m in row] for row in materials])

    phi1 = np.ones((ny, nx))
    phi2 = np.ones((ny, nx))
    k_old = 1.0
    result = EigenvalueResult(keff=k_old)
    result.keff_history.append(k_old)

    for iteration in range(max_iter):
        S1 = ((nuSigma_f1 * phi1 + nuSigma_f2 * phi2) / k_old).flatten()
        S2 = np.zeros(nx * ny)
        phi1_new, phi2_new, inner_it, _ = solve_2d_diffusion_2g_sor(
            geom, S1, S2, omega=omega, tol=inner_tol,
            max_iter=inner_max_iter,
            initial_guess1=phi1.copy(), initial_guess2=phi2.copy()
        )
        result.inner_iterations.append(inner_it)

        if not np.all(np.isfinite(phi1_new)):
            phi1_new = np.ones((ny, nx))
        if not np.all(np.isfinite(phi2_new)):
            phi2_new = np.ones((ny, nx))

        numerator = np.sum(nuSigma_f1 * phi1_new + nuSigma_f2 * phi2_new)
        denominator = np.sum(nuSigma_f1 * phi1 + nuSigma_f2 * phi2)
        k_new = k_old * numerator / denominator if denominator > 0 else k_old
        if not np.isfinite(k_new):
            k_new = k_old

        phi_max = max(np.max(phi1_new), np.max(phi2_new))
        if phi_max > 1e-30:
            phi1_new = phi1_new / phi_max
            phi2_new = phi2_new / phi_max
        else:
            phi1_new = np.ones((ny, nx))
            phi2_new = np.ones((ny, nx))

        k_error = abs(k_new - k_old) / k_old if k_old > 0 else 1.0
        flux_error1 = np.max(np.abs(phi1_new - phi1) / np.maximum(phi1, 1e-20))
        flux_error2 = np.max(np.abs(phi2_new - phi2) / np.maximum(phi2, 1e-20))
        flux_error = max(flux_error1, flux_error2)
        if not np.isfinite(k_error):
            k_error = 1.0
        if not np.isfinite(flux_error):
            flux_error = 1.0

        result.keff_history.append(k_new)
        result.flux_changes.append(flux_error)

        if k_error < k_tol and flux_error < flux_tol:
            phi1, phi2 = phi1_new, phi2_new
            k_old = k_new
            result.converged = True
            result.n_iterations = iteration + 1
            break

        phi1, phi2 = phi1_new, phi2_new
        k_old = k_new
        result.n_iterations = iteration + 1

    result.keff = k_old
    result.phi1 = phi1
    result.phi2 = phi2
    return result
