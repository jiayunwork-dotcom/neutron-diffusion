import numpy as np
from typing import List, Tuple, Optional, Callable
from copy import deepcopy
from .geometry import Geometry1D, Geometry2D, Region1D, Region2D, BoundaryCondition
from .materials import PRESET_MATERIALS_1G, PRESET_MATERIALS_2G, Material1G, Material2G
from .eigenvalue import (
    criticality_1d_1g, criticality_2d_1g,
    criticality_1d_2g, criticality_2d_2g
)


def reflector_thickness_sensitivity_1d(
    fuel_width: float,
    reflector_thickness_values: List[float],
    fuel_material: str = "UO2燃料(3%)",
    reflector_material: str = "石墨反射层",
    dx: float = 0.5,
    ngroups: str = "1g",
) -> Tuple[List[float], List[float]]:
    keff_values = []
    for rt in reflector_thickness_values:
        x0 = 0.0
        x1 = rt
        x2 = rt + fuel_width
        x3 = x2 + rt
        regions = [
            Region1D(x0, x1, reflector_material),
            Region1D(x1, x2, fuel_material),
            Region1D(x2, x3, reflector_material),
        ]
        geom = Geometry1D(regions=regions, dx=dx, material_mode=ngroups)
        if ngroups == "1g":
            result = criticality_1d_1g(geom)
        else:
            result = criticality_1d_2g(geom)
        keff_values.append(result.keff)
    return reflector_thickness_values, keff_values


def find_critical_size_1d(
    reflector_thickness: float,
    size_min: float,
    size_max: float,
    fuel_material: str = "UO2燃料(3%)",
    reflector_material: str = "石墨反射层",
    dx: float = 0.5,
    ngroups: str = "1g",
    tol: float = 1e-4,
    max_iter: int = 50,
) -> Tuple[float, List[float], List[float]]:
    sizes = []
    keffs = []
    a, b = size_min, size_max
    critical_size = (a + b) / 2.0
    for _ in range(max_iter):
        mid = (a + b) / 2.0
        rt = reflector_thickness
        x0 = 0.0
        x1 = rt
        x2 = rt + mid
        x3 = x2 + rt
        regions = [
            Region1D(x0, x1, reflector_material),
            Region1D(x1, x2, fuel_material),
            Region1D(x2, x3, reflector_material),
        ]
        geom = Geometry1D(regions=regions, dx=dx, material_mode=ngroups)
        if ngroups == "1g":
            result = criticality_1d_1g(geom)
        else:
            result = criticality_1d_2g(geom)
        keff = result.keff
        sizes.append(mid)
        keffs.append(keff)
        if abs(keff - 1.0) < tol:
            critical_size = mid
            break
        if keff < 1.0:
            a = mid
        else:
            b = mid
        critical_size = mid
    return critical_size, sizes, keffs
