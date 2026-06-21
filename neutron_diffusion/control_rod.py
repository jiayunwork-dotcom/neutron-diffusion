import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from copy import deepcopy
from .geometry import Geometry1D, Geometry2D, Region1D, Region2D
from .eigenvalue import (
    criticality_1d_1g, criticality_2d_1g,
    criticality_1d_2g, criticality_2d_2g, EigenvalueResult
)


@dataclass
class ControlRodResult:
    keff_out: float
    keff_in: float
    reactivity_pcm: float
    step_results: List[Tuple[int, float, float]] = field(default_factory=list)


def compute_reactivity(keff_out: float, keff_in: float) -> float:
    if keff_out <= 0 or keff_in <= 0:
        return 0.0
    rho = (keff_out - keff_in) / (keff_out * keff_in)
    return rho * 1e5


def control_rod_1d(
    geom: Geometry1D,
    rod_region: Tuple[float, float],
    ngroups: str = "1g",
    omega: float = 1.5,
) -> ControlRodResult:
    x_start, x_end = rod_region

    if ngroups == "1g":
        result_out = criticality_1d_1g(geom)
    else:
        result_out = criticality_1d_2g(geom)

    rod_regions = []
    for r in geom.regions:
        overlap_start = max(r.x_start, x_start)
        overlap_end = min(r.x_end, x_end)
        if overlap_start < overlap_end:
            if r.x_start < overlap_start:
                rod_regions.append(Region1D(r.x_start, overlap_start, r.material_name))
            rod_regions.append(Region1D(overlap_start, overlap_end, "B4C控制棒"))
            if overlap_end < r.x_end:
                rod_regions.append(Region1D(overlap_end, r.x_end, r.material_name))
        else:
            rod_regions.append(Region1D(r.x_start, r.x_end, r.material_name))

    geom_rod = deepcopy(geom)
    geom_rod.regions = rod_regions

    if ngroups == "1g":
        result_in = criticality_1d_1g(geom_rod)
    else:
        result_in = criticality_1d_2g(geom_rod)

    rho = compute_reactivity(result_out.keff, result_in.keff)
    return ControlRodResult(
        keff_out=result_out.keff,
        keff_in=result_in.keff,
        reactivity_pcm=rho
    )


def control_rod_2d(
    geom: Geometry2D,
    rod_regions_list: List[Tuple[float, float, float, float]],
    ngroups: str = "1g",
    omega: float = 1.5,
) -> ControlRodResult:
    if ngroups == "1g":
        result_out = criticality_2d_1g(geom, omega=omega)
    else:
        result_out = criticality_2d_2g(geom, omega=omega)

    new_regions = []
    for r in geom.regions:
        current_x = r.x_start
        current_y = r.y_start
        is_rod = False
        for (rx_s, rx_e, ry_s, ry_e) in rod_regions_list:
            if (r.x_start >= rx_s and r.x_end <= rx_e and
                r.y_start >= ry_s and r.y_end <= ry_e):
                is_rod = True
                break
        if is_rod:
            new_regions.append(Region2D(r.x_start, r.x_end, r.y_start, r.y_end, "B4C控制棒"))
        else:
            new_regions.append(Region2D(r.x_start, r.x_end, r.y_start, r.y_end, r.material_name))

    for (rx_s, rx_e, ry_s, ry_e) in rod_regions_list:
        new_regions.append(Region2D(rx_s, rx_e, ry_s, ry_e, "B4C控制棒"))

    geom_rod = deepcopy(geom)
    geom_rod.regions = new_regions

    if ngroups == "1g":
        result_in = criticality_2d_1g(geom_rod, omega=omega)
    else:
        result_in = criticality_2d_2g(geom_rod, omega=omega)

    rho = compute_reactivity(result_out.keff, result_in.keff)
    return ControlRodResult(
        keff_out=result_out.keff,
        keff_in=result_in.keff,
        reactivity_pcm=rho
    )


def control_rod_1d_step_insertion(
    geom: Geometry1D,
    rod_x_start: float,
    rod_x_end: float,
    n_steps: int = 10,
    ngroups: str = "1g",
) -> ControlRodResult:
    if ngroups == "1g":
        result_out = criticality_1d_1g(geom)
    else:
        result_out = criticality_1d_2g(geom)

    step_results = []
    total_length = rod_x_end - rod_x_start
    keff_prev = result_out.keff

    for step in range(1, n_steps + 1):
        current_end = rod_x_start + total_length * step / n_steps
        rod_regions = []
        for r in geom.regions:
            overlap_start = max(r.x_start, rod_x_start)
            overlap_end = min(r.x_end, current_end)
            if overlap_start < overlap_end:
                if r.x_start < overlap_start:
                    rod_regions.append(Region1D(r.x_start, overlap_start, r.material_name))
                rod_regions.append(Region1D(overlap_start, overlap_end, "B4C控制棒"))
                if overlap_end < r.x_end:
                    rod_regions.append(Region1D(overlap_end, r.x_end, r.material_name))
            else:
                rod_regions.append(Region1D(r.x_start, r.x_end, r.material_name))

        geom_step = deepcopy(geom)
        geom_step.regions = rod_regions

        if ngroups == "1g":
            result_step = criticality_1d_1g(geom_step)
        else:
            result_step = criticality_1d_2g(geom_step)

        rho_integral = compute_reactivity(result_out.keff, result_step.keff)
        rho_diff = compute_reactivity(keff_prev, result_step.keff)
        step_results.append((step, rho_integral, rho_diff))
        keff_prev = result_step.keff

    rho_total = step_results[-1][1] if step_results else 0.0
    return ControlRodResult(
        keff_out=result_out.keff,
        keff_in=keff_prev,
        reactivity_pcm=rho_total,
        step_results=step_results
    )
