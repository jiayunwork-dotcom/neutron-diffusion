import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List
from .geometry import Geometry2D, Region2D, BoundaryCondition
from .eigenvalue import criticality_2d_2g, EigenvalueResult


IAEA_2G_LWR_KEFF_REF = 1.02910


def build_iaea_benchmark_geometry(dx: float = 1.0, dy: float = 1.0) -> Geometry2D:
    size = 170.0
    regions = []

    regions.append(Region2D(0.0, 100.0, 0.0, 100.0, "IAEA燃料A"))
    regions.append(Region2D(100.0, 170.0, 0.0, 100.0, "IAEA燃料B"))
    regions.append(Region2D(0.0, 100.0, 100.0, 170.0, "IAEA燃料B"))
    regions.append(Region2D(100.0, 170.0, 100.0, 170.0, "石墨反射层"))

    bc = BoundaryCondition(
        left="reflective",
        right="vacuum",
        bottom="reflective",
        top="vacuum"
    )
    geom = Geometry2D(
        regions=regions,
        dx=dx,
        dy=dy,
        bc=bc,
        material_mode="2g"
    )
    return geom


@dataclass
class BenchmarkResult:
    keff_calc: float
    keff_ref: float
    keff_error_pcm: float
    converged: bool
    n_iterations: int
    phi1: np.ndarray = field(default_factory=lambda: np.array([]))
    phi2: np.ndarray = field(default_factory=lambda: np.array([]))
    x_centers: np.ndarray = field(default_factory=lambda: np.array([]))
    y_centers: np.ndarray = field(default_factory=lambda: np.array([]))


def run_iaea_benchmark(
    dx: float = 5.0,
    dy: float = 5.0,
    omega: float = 1.0,
    inner_tol: float = 1e-6,
    inner_max_iter: int = 2000,
    k_tol: float = 1e-5,
    flux_tol: float = 1e-4,
    max_iter: int = 500,
) -> BenchmarkResult:
    geom = build_iaea_benchmark_geometry(dx, dy)
    result: EigenvalueResult = criticality_2d_2g(
        geom,
        omega=omega,
        inner_tol=inner_tol,
        inner_max_iter=inner_max_iter,
        k_tol=k_tol,
        flux_tol=flux_tol,
        max_iter=max_iter
    )
    keff_error_pcm = abs(result.keff - IAEA_2G_LWR_KEFF_REF) / IAEA_2G_LWR_KEFF_REF * 1e5

    _, _, x_centers, y_centers, _ = geom.build_mesh()

    return BenchmarkResult(
        keff_calc=result.keff,
        keff_ref=IAEA_2G_LWR_KEFF_REF,
        keff_error_pcm=keff_error_pcm,
        converged=result.converged,
        n_iterations=result.n_iterations,
        phi1=result.phi1 if result.phi1 is not None else np.array([]),
        phi2=result.phi2 if result.phi2 is not None else np.array([]),
        x_centers=x_centers,
        y_centers=y_centers,
    )
