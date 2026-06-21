import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Tuple, Optional
from .geometry import Geometry1D, Geometry2D
from .eigenvalue import EigenvalueResult


def plot_1d_flux(
    x_centers: np.ndarray,
    phi: np.ndarray,
    phi2: Optional[np.ndarray] = None,
    title: str = "中子通量分布",
    xlabel: str = "x (cm)",
    ylabel: str = "归一化通量 φ",
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_centers, phi, "b-", linewidth=2, label="热群通量 φ" if phi2 is not None else "通量 φ")
    if phi2 is not None:
        ax.plot(x_centers, phi2, "r--", linewidth=2, label="快群通量 φ₁")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_2d_flux(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    phi: np.ndarray,
    title: str = "中子通量热力图",
    cmap: str = "viridis",
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    extent = [x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]]
    im = ax.imshow(phi, origin="lower", extent=extent, aspect="auto", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("归一化通量 φ", fontsize=12)
    ax.set_xlabel("x (cm)", fontsize=12)
    ax.set_ylabel("y (cm)", fontsize=12)
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_2d_contour(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    phi: np.ndarray,
    title: str = "中子通量等值线图",
    n_levels: int = 20,
    cmap: str = "viridis",
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    X, Y = np.meshgrid(x_centers, y_centers)
    cs = ax.contourf(X, Y, phi, levels=n_levels, cmap=cmap)
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("归一化通量 φ", fontsize=12)
    ax.contour(X, Y, phi, levels=n_levels, colors="k", linewidths=0.5, alpha=0.5)
    ax.set_xlabel("x (cm)", fontsize=12)
    ax.set_ylabel("y (cm)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_convergence(
    keff_history: List[float],
    title: str = "keff收敛曲线",
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    iterations = list(range(len(keff_history)))
    ax.plot(iterations, keff_history, "bo-", linewidth=2, markersize=4)
    ax.set_xlabel("外迭代次数", fontsize=12)
    ax.set_ylabel("keff", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    if len(keff_history) > 1:
        ax.axhline(y=keff_history[-1], color="r", linestyle="--", alpha=0.7, label=f"keff = {keff_history[-1]:.6f}")
        ax.legend(fontsize=11)
    fig.tight_layout()
    return fig


def plot_power_distribution_1d(
    x_centers: np.ndarray,
    power: np.ndarray,
    title: str = "归一化功率分布",
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_centers, power, "g-", linewidth=2)
    ax.set_xlabel("x (cm)", fontsize=12)
    ax.set_ylabel("归一化相对功率", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    peak_factor = np.max(power) / np.mean(power) if np.mean(power) > 0 else 1.0
    ax.axhline(y=np.mean(power), color="r", linestyle="--", alpha=0.7, label=f"平均功率 = {np.mean(power):.3f}")
    ax.axhline(y=np.max(power), color="m", linestyle="--", alpha=0.7, label=f"最大功率 = {np.max(power):.3f}")
    ax.text(0.02, 0.95, f"功率峰因子 = {peak_factor:.3f}", transform=ax.transAxes,
            fontsize=12, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend(fontsize=11)
    fig.tight_layout()
    return fig


def plot_power_distribution_2d(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    power: np.ndarray,
    title: str = "归一化功率分布热力图",
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    extent = [x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]]
    im = ax.imshow(power, origin="lower", extent=extent, aspect="auto", cmap="hot")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("归一化相对功率", fontsize=12)
    ax.set_xlabel("x (cm)", fontsize=12)
    ax.set_ylabel("y (cm)", fontsize=12)
    peak_factor = np.max(power) / np.mean(power) if np.mean(power) > 0 else 1.0
    ax.set_title(f"{title}\n功率峰因子 = {peak_factor:.3f}", fontsize=14)
    fig.tight_layout()
    return fig


def plot_sensitivity(
    param_values: List[float],
    keff_values: List[float],
    param_name: str = "参数",
    title: str = "参数敏感性分析",
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(param_values, keff_values, "bo-", linewidth=2, markersize=6)
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.7, label="keff = 1.0 (临界)")
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel("keff", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_control_rod_worth(
    step_results: List[Tuple[int, float, float]],
    title: str = "控制棒价值曲线",
) -> Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    steps = [s[0] for s in step_results]
    rho_int = [s[1] for s in step_results]
    rho_diff = [s[2] for s in step_results]

    ax1.plot(steps, rho_int, "b-o", linewidth=2, markersize=4)
    ax1.set_xlabel("插入步数", fontsize=12)
    ax1.set_ylabel("积分反应性 (pcm)", fontsize=12)
    ax1.set_title("积分控制棒价值", fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.bar(steps, rho_diff, color="g", alpha=0.7)
    ax2.set_xlabel("插入步数", fontsize=12)
    ax2.set_ylabel("微分反应性 (pcm)", fontsize=12)
    ax2.set_title("微分控制棒价值", fontsize=14)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    return fig


def plot_1d_profile(
    coords: np.ndarray,
    values: np.ndarray,
    title: str = "通量剖面",
    xlabel: str = "位置 (cm)",
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(coords, values, "b-", linewidth=2)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("归一化通量", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_kinetics_power(
    time: np.ndarray,
    power_norm: np.ndarray,
    title: str = "归一化功率 P(t)/P0 随时间变化",
    log_scale: bool = False,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, power_norm, "b-", linewidth=2, label="P(t)/P₀")
    ax.set_xlabel("时间 t (s)", fontsize=12)
    ax.set_ylabel("归一化功率 P(t)/P₀", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    if log_scale:
        ax.set_yscale("log")
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="初始功率")
    ax.legend(fontsize=11)
    fig.tight_layout()
    return fig


def plot_kinetics_reactivity(
    time: np.ndarray,
    reactivity: np.ndarray,
    title: str = "反应性 ρ(t) 随时间变化",
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, reactivity, "m-", linewidth=2, label="ρ(t)")
    ax.set_xlabel("时间 t (s)", fontsize=12)
    ax.set_ylabel("反应性 ρ", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.0, color="k", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_kinetics_precursors(
    time: np.ndarray,
    precursors: np.ndarray,
    lambda_i: np.ndarray,
    title: str = "各组缓发中子先驱核浓度随时间变化",
    normalize: bool = True,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["b", "g", "r", "c", "m", "y"]
    n_groups = precursors.shape[1]
    for i in range(n_groups):
        c = precursors[:, i]
        if normalize and c[0] > 0:
            c = c / c[0]
        label = f"组{i+1} (λ={lambda_i[i]:.3f} s⁻¹)"
        ax.plot(time, c, color=colors[i % len(colors)], linewidth=1.5, label=label)
    ax.set_xlabel("时间 t (s)", fontsize=12)
    ax.set_ylabel("归一化先驱核浓度 Cᵢ(t)/Cᵢ(0)" if normalize else "先驱核浓度 Cᵢ(t)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_kinetics_all(
    time: np.ndarray,
    power_norm: np.ndarray,
    reactivity: np.ndarray,
    precursors: np.ndarray,
    lambda_i: np.ndarray,
) -> Figure:
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    axes[0].plot(time, power_norm, "b-", linewidth=2)
    axes[0].set_ylabel("P(t)/P₀", fontsize=12)
    axes[0].set_title("归一化功率", fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=1.0, color="r", linestyle="--", alpha=0.5)

    axes[1].plot(time, reactivity, "m-", linewidth=2)
    axes[1].set_ylabel("ρ(t)", fontsize=12)
    axes[1].set_title("反应性", fontsize=13)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.0, color="k", linestyle="--", alpha=0.5)

    colors = ["b", "g", "r", "c", "m", "y"]
    n_groups = precursors.shape[1]
    for i in range(n_groups):
        c = precursors[:, i]
        if c[0] > 0:
            c = c / c[0]
        label = f"组{i+1} (λ={lambda_i[i]:.3f})"
        axes[2].plot(time, c, color=colors[i % len(colors)], linewidth=1.5, label=label)
    axes[2].set_xlabel("时间 t (s)", fontsize=12)
    axes[2].set_ylabel("Cᵢ(t)/Cᵢ(0)", fontsize=12)
    axes[2].set_title("归一化先驱核浓度", fontsize=13)
    axes[2].legend(fontsize=8, loc="best")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_keff_convergence_mc(
    keff_history: List[float],
    n_discard: int = 0,
    keff_mean: Optional[float] = None,
    keff_std: Optional[float] = None,
    window: int = 5,
    title: str = "蒙特卡洛 keff 收敛曲线",
) -> Figure:
    from .monte_carlo import moving_average

    fig, ax = plt.subplots(figsize=(10, 6))
    generations = list(range(1, len(keff_history) + 1))

    ax.plot(generations, keff_history, "bo", markersize=4, alpha=0.6, label="每代 keff")

    ma = moving_average(keff_history, window=window)
    ax.plot(generations, ma, "r-", linewidth=2, label=f"移动平均 (窗口={window})")

    if n_discard > 0 and n_discard < len(keff_history):
        ax.axvline(x=n_discard + 0.5, color="k", linestyle="--", alpha=0.7,
                   label=f"丢弃前 {n_discard} 代")
        ax.axvspan(0.5, n_discard + 0.5, alpha=0.1, color="gray")

    if keff_mean is not None:
        ax.axhline(y=keff_mean, color="g", linestyle="--", linewidth=2,
                   label=f"平均 keff = {keff_mean:.6f}")
        if keff_std is not None:
            ax.axhline(y=keff_mean + keff_std, color="g", linestyle=":", alpha=0.5)
            ax.axhline(y=keff_mean - keff_std, color="g", linestyle=":", alpha=0.5,
                       label=f"±σ = ±{keff_std:.6f}")

    ax.set_xlabel("代际", fontsize=12)
    ax.set_ylabel("keff", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_flux_comparison(
    x_mc: np.ndarray,
    flux_mc: np.ndarray,
    x_det: np.ndarray,
    flux_det: np.ndarray,
    title: str = "蒙特卡洛 vs 扩散方程 通量对比",
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))

    flux_det_norm = flux_det.copy()
    max_det = np.max(flux_det)
    if max_det > 0:
        flux_det_norm = flux_det / max_det

    flux_mc_norm = flux_mc.copy()
    max_mc = np.max(flux_mc)
    if max_mc > 0:
        flux_mc_norm = flux_mc / max_mc

    ax.plot(x_det, flux_det_norm, "b-", linewidth=2.5, label="扩散方程 (确定性)")
    ax.plot(x_mc, flux_mc_norm, "ro", markersize=5, alpha=0.7, label="蒙特卡洛")

    ax.set_xlabel("x (cm)", fontsize=12)
    ax.set_ylabel("归一化通量 φ", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_relative_error(
    x: np.ndarray,
    rel_error: np.ndarray,
    title: str = "通量相对误差分布",
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x, rel_error, width=(x[1] - x[0]) * 0.8, color="orange", alpha=0.7, edgecolor="k")

    mean_err = np.mean(rel_error)
    max_err = np.max(rel_error)
    ax.axhline(y=mean_err, color="r", linestyle="--", linewidth=2,
               label=f"平均误差 = {mean_err:.2f}%")
    ax.axhline(y=max_err, color="m", linestyle=":", linewidth=2,
               label=f"最大误差 = {max_err:.2f}%")

    ax.set_xlabel("x (cm)", fontsize=12)
    ax.set_ylabel("相对误差 (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_collision_histogram(
    x_centers: np.ndarray,
    collision_count: np.ndarray,
    title: str = "中子碰撞次数分布",
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))

    bin_width = x_centers[1] - x_centers[0] if len(x_centers) > 1 else 1.0
    ax.bar(x_centers, collision_count, width=bin_width * 0.8, color="steelblue", alpha=0.7, edgecolor="k")

    ax.set_xlabel("x (cm)", fontsize=12)
    ax.set_ylabel("碰撞次数", fontsize=12)
    ax.set_title(f"{title}\n总碰撞次数: {int(np.sum(collision_count)):,}", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig
