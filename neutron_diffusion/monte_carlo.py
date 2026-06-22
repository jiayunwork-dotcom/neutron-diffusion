import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
from .geometry import Geometry1D
from .materials import Material1G, get_preset_1g


@dataclass
class Neutron:
    x: float
    direction: int
    weight: float = 1.0
    alive: bool = True


@dataclass
class MCReaction:
    reaction_type: str
    x: float
    weight: float


@dataclass
class MonteCarloResult:
    keff_history: List[float] = field(default_factory=list)
    keff_mean: float = 0.0
    keff_std: float = 0.0
    keff_ci_95: Tuple[float, float] = (0.0, 0.0)
    flux_bins: np.ndarray = field(default_factory=lambda: np.array([]))
    flux_centers: np.ndarray = field(default_factory=lambda: np.array([]))
    flux_mc: np.ndarray = field(default_factory=lambda: np.array([]))
    track_length: np.ndarray = field(default_factory=lambda: np.array([]))
    collision_count: np.ndarray = field(default_factory=lambda: np.array([]))
    n_generations: int = 0
    n_discard: int = 0
    n_neutrons_per_gen: int = 0
    total_neutrons_tracked: int = 0
    total_collisions: int = 0
    fission_sites: List[float] = field(default_factory=list)
    seed: Optional[int] = None

    @property
    def effective_generations(self) -> int:
        return max(0, self.n_generations - self.n_discard)

    @property
    def keff_relative_error(self) -> float:
        if self.keff_mean > 0:
            return self.keff_std / self.keff_mean
        return 0.0


def _get_region_idx_at_x(geom: Geometry1D, x: float) -> int:
    for i, region in enumerate(geom.regions):
        if i == len(geom.regions) - 1:
            if region.x_start - 1e-12 <= x <= region.x_end + 1e-12:
                return i
        else:
            if region.x_start - 1e-12 <= x < region.x_end - 1e-12:
                return i
    return -1


def _get_material_at_x(geom: Geometry1D, x: float) -> Optional[Material1G]:
    idx = _get_region_idx_at_x(geom, x)
    if idx >= 0:
        return get_preset_1g(geom.regions[idx].material_name)
    return None


def _find_next_boundary(geom: Geometry1D, x: float, direction: int) -> float:
    idx = _get_region_idx_at_x(geom, x)
    if idx < 0:
        if x < geom.x_min:
            return geom.x_min
        return geom.x_max

    if direction > 0:
        return geom.regions[idx].x_end
    else:
        return geom.regions[idx].x_start


def _sample_free_path(Sigma_t: float, rng: np.random.Generator) -> float:
    if Sigma_t <= 0:
        return 1e10
    xi = rng.random()
    while xi <= 0.0:
        xi = rng.random()
    return -np.log(xi) / Sigma_t


def _sample_reaction_type(
    mat: Material1G, rng: np.random.Generator
) -> str:
    Sigma_t = mat.Sigma_a + mat.Sigma_s
    if Sigma_t <= 0:
        return "absorption"

    xi = rng.random()
    p_scat = mat.Sigma_s / Sigma_t

    if xi < p_scat:
        return "scattering"
    else:
        if mat.Sigma_a > 0:
            p_fission = mat.Sigma_f / mat.Sigma_a
            if rng.random() < p_fission:
                return "fission"
            else:
                return "absorption"
        else:
            return "absorption"


def _sample_direction(rng: np.random.Generator) -> int:
    return 1 if rng.random() >= 0.5 else -1


def _sample_fission_neutrons(nu: float, rng: np.random.Generator) -> int:
    n_integer = int(nu)
    remainder = nu - n_integer
    if rng.random() < remainder:
        n_integer += 1
    return n_integer


def _accumulate_track_length(
    x_start: float,
    x_end: float,
    bin_edges: np.ndarray,
    track_lengths: np.ndarray,
    weight: float = 1.0,
) -> None:
    if x_start == x_end:
        return

    if x_start > x_end:
        x_start, x_end = x_end, x_start

    x_min_bin = bin_edges[0]
    x_max_bin = bin_edges[-1]

    x_start_clamped = max(x_start, x_min_bin)
    x_end_clamped = min(x_end, x_max_bin)

    if x_start_clamped >= x_end_clamped:
        return

    idx_start = np.searchsorted(bin_edges, x_start_clamped, side="right") - 1
    idx_end = np.searchsorted(bin_edges, x_end_clamped, side="right") - 1

    idx_start = max(0, min(idx_start, len(track_lengths) - 1))
    idx_end = max(0, min(idx_end, len(track_lengths) - 1))

    if idx_start == idx_end:
        track_lengths[idx_start] += (x_end_clamped - x_start_clamped) * weight
    else:
        track_lengths[idx_start] += (bin_edges[idx_start + 1] - x_start_clamped) * weight
        for i in range(idx_start + 1, idx_end):
            track_lengths[i] += (bin_edges[i + 1] - bin_edges[i]) * weight
        track_lengths[idx_end] += (x_end_clamped - bin_edges[idx_end]) * weight


def _track_neutron(
    neutron: Neutron,
    geom: Geometry1D,
    rng: np.random.Generator,
    bin_edges: Optional[np.ndarray] = None,
    track_lengths: Optional[np.ndarray] = None,
    track_collisions: bool = True,
) -> Tuple[List[MCReaction], List[float], str]:
    collisions: List[MCReaction] = []
    fission_sites: List[float] = []
    max_collisions = 10000
    n_collisions = 0
    end_reason = "absorption"

    while neutron.alive and n_collisions < max_collisions:
        x_before = neutron.x

        mat = _get_material_at_x(geom, neutron.x)
        if mat is None:
            neutron.alive = False
            end_reason = "leakage"
            break

        Sigma_t = mat.Sigma_a + mat.Sigma_s
        path_length = _sample_free_path(Sigma_t, rng)

        next_boundary = _find_next_boundary(geom, neutron.x, neutron.direction)
        dist_to_boundary = abs(next_boundary - neutron.x)

        if path_length < dist_to_boundary:
            new_x = neutron.x + neutron.direction * path_length

            if bin_edges is not None and track_lengths is not None:
                _accumulate_track_length(neutron.x, new_x, bin_edges, track_lengths, neutron.weight)

            neutron.x = new_x
            n_collisions += 1

            reaction = _sample_reaction_type(mat, rng)

            if track_collisions:
                collisions.append(MCReaction(reaction, neutron.x, neutron.weight))

            if reaction == "absorption":
                neutron.alive = False
                end_reason = "absorption"
            elif reaction == "scattering":
                neutron.direction = _sample_direction(rng)
            elif reaction == "fission":
                neutron.alive = False
                end_reason = "fission"
                if mat.Sigma_f > 0:
                    n_new = _sample_fission_neutrons(mat.nu, rng)
                    for _ in range(n_new):
                        fission_sites.append(neutron.x)
        else:
            new_x = next_boundary

            if bin_edges is not None and track_lengths is not None:
                _accumulate_track_length(neutron.x, new_x, bin_edges, track_lengths, neutron.weight)

            neutron.x = new_x

            if neutron.x <= geom.x_min + 1e-12:
                if geom.bc.left == "reflective":
                    neutron.x = geom.x_min
                    neutron.direction = 1
                elif geom.bc.left == "zero_flux" or geom.bc.left == "vacuum":
                    neutron.alive = False
                    end_reason = "leakage"
                    break
                else:
                    neutron.alive = False
                    end_reason = "leakage"
                    break
            elif neutron.x >= geom.x_max - 1e-12:
                if geom.bc.right == "reflective":
                    neutron.x = geom.x_max
                    neutron.direction = -1
                elif geom.bc.right == "zero_flux" or geom.bc.right == "vacuum":
                    neutron.alive = False
                    end_reason = "leakage"
                    break
                else:
                    neutron.alive = False
                    end_reason = "leakage"
                    break
            else:
                eps = 1e-10 * (geom.x_max - geom.x_min)
                if neutron.direction > 0:
                    neutron.x = next_boundary + eps
                else:
                    neutron.x = next_boundary - eps

    return collisions, fission_sites, end_reason


def _initialize_source(
    geom: Geometry1D,
    n_neutrons: int,
    rng: np.random.Generator,
) -> List[Neutron]:
    neutrons: List[Neutron] = []
    x_min, x_max = geom.x_min, geom.x_max

    for _ in range(n_neutrons):
        x = x_min + rng.random() * (x_max - x_min)
        direction = _sample_direction(rng)
        neutrons.append(Neutron(x=x, direction=direction, weight=1.0, alive=True))

    return neutrons


def _initialize_source_from_sites(
    fission_sites: List[float],
    n_neutrons: int,
    rng: np.random.Generator,
) -> List[Neutron]:
    neutrons: List[Neutron] = []

    if not fission_sites:
        return neutrons

    sites = np.array(fission_sites)
    n_sites = len(sites)

    for i in range(n_neutrons):
        idx = i % n_sites if n_sites > 0 else 0
        x = sites[idx]
        direction = _sample_direction(rng)
        neutrons.append(Neutron(x=x, direction=direction, weight=1.0, alive=True))

    return neutrons


def _setup_flux_bins(geom: Geometry1D, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    x_min, x_max = geom.x_min, geom.x_max
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return bin_edges, bin_centers


def _accumulate_flux_collisions(
    collisions: List[MCReaction],
    bin_edges: np.ndarray,
    collision_counts: np.ndarray,
) -> np.ndarray:
    for col in collisions:
        idx = np.searchsorted(bin_edges, col.x, side="right") - 1
        if 0 <= idx < len(collision_counts):
            collision_counts[idx] += col.weight
    return collision_counts


def _get_bin_Sigma_t(
    geom: Geometry1D,
    bin_centers: np.ndarray,
) -> np.ndarray:
    Sigma_t_bins = np.zeros_like(bin_centers)
    for i, xc in enumerate(bin_centers):
        mat = _get_material_at_x(geom, xc)
        if mat is not None:
            Sigma_t_bins[i] = mat.Sigma_a + mat.Sigma_s
        else:
            Sigma_t_bins[i] = 1e-10
    return Sigma_t_bins


def run_monte_carlo_1d(
    geom: Geometry1D,
    n_neutrons_per_gen: int = 10000,
    n_generations: int = 100,
    n_discard: int = 20,
    n_flux_bins: int = 50,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> MonteCarloResult:
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    result = MonteCarloResult(
        n_generations=n_generations,
        n_discard=n_discard,
        n_neutrons_per_gen=n_neutrons_per_gen,
        seed=seed,
    )

    bin_edges, bin_centers = _setup_flux_bins(geom, n_flux_bins)
    track_lengths = np.zeros(n_flux_bins, dtype=np.float64)
    collision_counts = np.zeros(n_flux_bins, dtype=np.float64)
    Sigma_t_bins = _get_bin_Sigma_t(geom, bin_centers)
    result.flux_bins = bin_edges
    result.flux_centers = bin_centers

    keff_per_gen: List[float] = []
    all_fission_sites: List[float] = []
    total_collisions = 0
    total_neutrons_tracked = 0

    neutrons = _initialize_source(geom, n_neutrons_per_gen, rng)

    for gen in range(n_generations):
        gen_fission_sites: List[float] = []
        gen_collisions: List[MCReaction] = []

        gen_track_lengths = np.zeros(n_flux_bins, dtype=np.float64) if gen >= n_discard else None

        for neutron in neutrons:
            if not neutron.alive:
                continue

            collisions, fission_sites, _ = _track_neutron(
                neutron, geom, rng,
                bin_edges=bin_edges if gen >= n_discard else None,
                track_lengths=gen_track_lengths,
            )
            gen_collisions.extend(collisions)
            gen_fission_sites.extend(fission_sites)

            total_collisions += len(collisions)
            total_neutrons_tracked += 1

        if gen >= n_discard and gen_track_lengths is not None:
            track_lengths += gen_track_lengths
            collision_counts = _accumulate_flux_collisions(gen_collisions, bin_edges, collision_counts)
            all_fission_sites.extend(gen_fission_sites)

        keff_gen = len(gen_fission_sites) / n_neutrons_per_gen if n_neutrons_per_gen > 0 else 0.0
        keff_per_gen.append(keff_gen)

        if progress_callback is not None:
            progress_callback(gen + 1, n_generations, keff_gen)

        if gen < n_generations - 1:
            if gen_fission_sites:
                neutrons = _initialize_source_from_sites(gen_fission_sites, n_neutrons_per_gen, rng)
            else:
                neutrons = _initialize_source(geom, n_neutrons_per_gen, rng)

    result.keff_history = keff_per_gen
    result.total_collisions = total_collisions
    result.total_neutrons_tracked = total_neutrons_tracked
    result.fission_sites = all_fission_sites
    result.collision_count = collision_counts
    result.track_length = track_lengths

    if result.effective_generations > 0:
        effective_keffs = keff_per_gen[n_discard:]
        result.keff_mean = float(np.mean(effective_keffs))
        result.keff_std = float(np.std(effective_keffs, ddof=1) / np.sqrt(result.effective_generations))
        ci = 1.96 * result.keff_std
        result.keff_ci_95 = (result.keff_mean - ci, result.keff_mean + ci)

    total_track_length = np.sum(track_lengths)
    if total_track_length > 0:
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        result.flux_mc = track_lengths / (total_track_length * bin_widths)
        max_flux = np.max(result.flux_mc)
        if max_flux > 0:
            result.flux_mc = result.flux_mc / max_flux

    return result


def compute_flux_relative_error(
    flux_mc: np.ndarray,
    flux_det: np.ndarray,
) -> np.ndarray:
    flux_det_normalized = flux_det.copy()
    max_det = np.max(flux_det)
    if max_det > 0:
        flux_det_normalized = flux_det / max_det

    max_mc = np.max(flux_mc)
    flux_mc_normalized = flux_mc.copy()
    if max_mc > 0:
        flux_mc_normalized = flux_mc / max_mc

    rel_error = np.zeros_like(flux_mc_normalized)
    mask = flux_det_normalized > 1e-10
    rel_error[mask] = np.abs(flux_mc_normalized[mask] - flux_det_normalized[mask]) / flux_det_normalized[mask] * 100.0

    return rel_error


def moving_average(data: List[float], window: int = 5) -> np.ndarray:
    if len(data) < window:
        return np.array(data, dtype=np.float64)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="same")
