import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional


@dataclass
class DelayedNeutronParams:
    beta_i: np.ndarray = field(default_factory=lambda: np.array([
        0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273
    ]))
    lambda_i: np.ndarray = field(default_factory=lambda: np.array([
        0.0127, 0.0317, 0.115, 0.311, 1.40, 3.87
    ]))

    @property
    def beta_total(self) -> float:
        return float(np.sum(self.beta_i))

    @property
    def n_groups(self) -> int:
        return len(self.beta_i)


def get_u235_params() -> DelayedNeutronParams:
    return DelayedNeutronParams(
        beta_i=np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273]),
        lambda_i=np.array([0.0127, 0.0317, 0.115, 0.311, 1.40, 3.87])
    )


def get_u238_params() -> DelayedNeutronParams:
    return DelayedNeutronParams(
        beta_i=np.array([0.000134, 0.001170, 0.001180, 0.002430, 0.000750, 0.000230]),
        lambda_i=np.array([0.0129, 0.0311, 0.134, 0.331, 1.26, 3.21])
    )


def get_pu239_params() -> DelayedNeutronParams:
    return DelayedNeutronParams(
        beta_i=np.array([0.000072, 0.000620, 0.000800, 0.001400, 0.000580, 0.000200]),
        lambda_i=np.array([0.0129, 0.0311, 0.082, 0.163, 0.84, 2.73])
    )


def reactivity_step(t: float, rho_step: float, t_insert: float) -> float:
    return rho_step if t >= t_insert else 0.0


def reactivity_linear(t: float, t_start: float, t_end: float, rho_start: float, rho_end: float) -> float:
    if t < t_start:
        return rho_start
    elif t > t_end:
        return rho_end
    else:
        frac = (t - t_start) / (t_end - t_start)
        return rho_start + frac * (rho_end - rho_start)


def reactivity_sinusoidal(t: float, amplitude: float, frequency: float, phase: float = 0.0, offset: float = 0.0) -> float:
    return offset + amplitude * np.sin(2.0 * np.pi * frequency * t + phase)


@dataclass
class KineticsResult:
    time: np.ndarray
    power: np.ndarray
    precursors: np.ndarray
    reactivity: np.ndarray
    Lambda: float
    beta_total: float
    beta_i: np.ndarray
    lambda_i: np.ndarray
    n_steps: int
    dt: float
    t_end: float

    @property
    def power_normalized(self) -> np.ndarray:
        return self.power / self.power[0] if self.power[0] > 0 else self.power

    @property
    def max_power_multiplier(self) -> float:
        return float(np.max(self.power_normalized))

    @property
    def steady_state_period(self) -> Optional[float]:
        if len(self.power_normalized) < 10:
            return None
        p = self.power_normalized
        idx_start = max(0, len(p) // 4)
        idx_end = len(p) - 1
        if p[idx_start] <= 0 or p[idx_end] <= 0 or p[idx_end] == p[idx_start]:
            return None
        ratio = p[idx_end] / p[idx_start]
        if ratio <= 0:
            return None
        dt = self.time[idx_end] - self.time[idx_start]
        return float(dt / np.log(ratio))

    @property
    def asymptotic_period(self) -> Optional[float]:
        if len(self.power_normalized) < 20:
            return None
        p = self.power_normalized
        tail_frac = 0.1
        n_tail = max(10, int(len(p) * tail_frac))
        p_tail = p[-n_tail:]
        t_tail = self.time[-n_tail:]
        if np.min(p_tail) <= 0:
            return None
        log_p = np.log(p_tail)
        coeffs = np.polyfit(t_tail, log_p, 1)
        slope = coeffs[0]
        if abs(slope) < 1e-10:
            return None
        return float(1.0 / slope)


def point_kinetics_rhs(
    state: np.ndarray,
    t: float,
    rho_func: Callable[[float], float],
    beta_i: np.ndarray,
    lambda_i: np.ndarray,
    beta_total: float,
    Lambda: float
) -> np.ndarray:
    P = state[0]
    C = state[1:]
    rho = rho_func(t)
    dPdt = ((rho - beta_total) / Lambda) * P + np.sum(lambda_i * C)
    dCdt = (beta_i / Lambda) * P - lambda_i * C
    return np.concatenate(([dPdt], dCdt))


def solve_point_kinetics_rk4(
    rho_func: Callable[[float], float],
    t_end: float = 10.0,
    dt: float = 0.001,
    Lambda: float = 1e-4,
    dn_params: Optional[DelayedNeutronParams] = None,
    P0: float = 1.0
) -> KineticsResult:
    if dn_params is None:
        dn_params = get_u235_params()
    beta_i = dn_params.beta_i
    lambda_i = dn_params.lambda_i
    beta_total = dn_params.beta_total
    n_groups = dn_params.n_groups

    n_steps = int(t_end / dt) + 1
    time = np.linspace(0.0, t_end, n_steps)

    C0 = (beta_i / Lambda) * P0 / lambda_i
    state = np.concatenate(([P0], C0))

    power_hist = np.zeros(n_steps)
    precursors_hist = np.zeros((n_steps, n_groups))
    reactivity_hist = np.zeros(n_steps)

    power_hist[0] = P0
    precursors_hist[0, :] = C0
    reactivity_hist[0] = rho_func(0.0)

    for i in range(n_steps - 1):
        t = time[i]
        k1 = point_kinetics_rhs(state, t, rho_func, beta_i, lambda_i, beta_total, Lambda)
        k2 = point_kinetics_rhs(state + 0.5 * dt * k1, t + 0.5 * dt, rho_func, beta_i, lambda_i, beta_total, Lambda)
        k3 = point_kinetics_rhs(state + 0.5 * dt * k2, t + 0.5 * dt, rho_func, beta_i, lambda_i, beta_total, Lambda)
        k4 = point_kinetics_rhs(state + dt * k3, t + dt, rho_func, beta_i, lambda_i, beta_total, Lambda)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        power_hist[i + 1] = state[0]
        precursors_hist[i + 1, :] = state[1:]
        reactivity_hist[i + 1] = rho_func(time[i + 1])

    return KineticsResult(
        time=time,
        power=power_hist,
        precursors=precursors_hist,
        reactivity=reactivity_hist,
        Lambda=Lambda,
        beta_total=beta_total,
        beta_i=beta_i,
        lambda_i=lambda_i,
        n_steps=n_steps,
        dt=dt,
        t_end=t_end
    )


def solve_point_kinetics_implicit_euler(
    rho_func: Callable[[float], float],
    t_end: float = 10.0,
    dt: float = 0.001,
    Lambda: float = 1e-4,
    dn_params: Optional[DelayedNeutronParams] = None,
    P0: float = 1.0
) -> KineticsResult:
    if dn_params is None:
        dn_params = get_u235_params()
    beta_i = dn_params.beta_i
    lambda_i = dn_params.lambda_i
    beta_total = dn_params.beta_total
    n_groups = dn_params.n_groups

    n_steps = int(t_end / dt) + 1
    time = np.linspace(0.0, t_end, n_steps)

    C0 = (beta_i / Lambda) * P0 / lambda_i
    P = P0
    C = C0.copy()

    power_hist = np.zeros(n_steps)
    precursors_hist = np.zeros((n_steps, n_groups))
    reactivity_hist = np.zeros(n_steps)

    power_hist[0] = P0
    precursors_hist[0, :] = C0
    reactivity_hist[0] = rho_func(0.0)

    for i in range(n_steps - 1):
        t_new = time[i + 1]
        rho_new = rho_func(t_new)

        A = 1.0 - dt * (rho_new - beta_total) / Lambda
        B_vec = dt * lambda_i
        D_vec = 1.0 + dt * lambda_i
        E_vec = dt * beta_i / Lambda

        denominator = A - np.sum(B_vec * E_vec / D_vec)
        P_new = (P + np.sum(B_vec * C / D_vec)) / denominator
        C_new = (C + E_vec * P_new) / D_vec

        P = P_new
        C = C_new

        power_hist[i + 1] = P
        precursors_hist[i + 1, :] = C
        reactivity_hist[i + 1] = rho_new

    return KineticsResult(
        time=time,
        power=power_hist,
        precursors=precursors_hist,
        reactivity=reactivity_hist,
        Lambda=Lambda,
        beta_total=beta_total,
        beta_i=beta_i,
        lambda_i=lambda_i,
        n_steps=n_steps,
        dt=dt,
        t_end=t_end
    )
