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


def _compute_keff_for_size(
    fuel_size: float,
    reflector_thickness: float,
    fuel_material: str,
    reflector_material: str,
    dx: float,
    ngroups: str,
) -> float:
    rt = reflector_thickness
    x0 = 0.0
    x1 = rt
    x2 = rt + fuel_size
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
    return result.keff


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
    auto_expand_range: bool = True,
) -> Tuple[float, List[float], List[float]]:
    sizes = []
    keffs = []
    computed = {}

    def eval_size(s: float) -> float:
        key = round(s / max(dx * 0.01, 1e-6))
        if key in computed:
            return computed[key]
        k = _compute_keff_for_size(s, reflector_thickness, fuel_material,
                                   reflector_material, dx, ngroups)
        computed[key] = k
        sizes.append(s)
        keffs.append(k)
        return k

    a, b = size_min, size_max

    if auto_expand_range:
        keff_a = eval_size(a)
        while keff_a > 1.0 and a > dx:
            a = max(dx, a / 2.0)
            keff_a = eval_size(a)

        keff_b = eval_size(b)
        expand_count = 0
        while keff_b < 1.0 and expand_count < 20:
            b = b * 1.5
            keff_b = eval_size(b)
            expand_count += 1

        if keff_a > 1.0 and a <= dx * 1.01:
            return a, sizes, keffs
        if keff_b < 1.0:
            return b, sizes, keffs
    else:
        keff_a = eval_size(a)
        keff_b = eval_size(b)

    critical_size = (a + b) / 2.0
    iter_count = 0
    while iter_count < max_iter and (b - a) > dx * 0.1:
        mid = (a + b) / 2.0
        keff_mid = eval_size(mid)
        critical_size = mid
        if abs(keff_mid - 1.0) < tol:
            break
        if keff_mid < 1.0:
            a = mid
            keff_a = keff_mid
        else:
            b = mid
            keff_b = keff_mid
        iter_count += 1
    return critical_size, sizes, keffs
