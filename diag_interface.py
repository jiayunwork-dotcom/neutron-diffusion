import numpy as np
import sys
sys.path.insert(0, '.')

from neutron_diffusion.geometry import Geometry1D, Region1D, BoundaryCondition
from neutron_diffusion.materials import get_preset_1g
from neutron_diffusion.eigenvalue import criticality_1d_1g
from neutron_diffusion.monte_carlo import (
    run_monte_carlo_1d, compute_flux_relative_error,
    _get_material_at_x, _get_region_idx_at_x, _find_next_boundary,
    _track_neutron, Neutron
)

print("=" * 70)
print("诊断：材料交界面处的通量偏差")
print("=" * 70)

# 构造: 石墨反射层[0,10] + UO2燃料[10,50] + 石墨反射层[50,60]
regions = [
    Region1D(0.0, 10.0, "石墨反射层"),
    Region1D(10.0, 50.0, "UO2燃料(3%)"),
    Region1D(50.0, 60.0, "石墨反射层"),
]
geom = Geometry1D(
    regions=regions,
    dx=0.5,
    bc=BoundaryCondition(left="vacuum", right="vacuum"),
    material_mode="1g"
)

print(f"\n几何: 反射层[0,10] + 燃料[10,50] + 反射层[50,60]")
print(f"边界条件: 左右真空")
print(f"网格间距: {geom.dx} cm")

# 1. 扩散方程求解
print("\n--- 扩散方程 ---")
det = criticality_1d_1g(geom, k_tol=1e-7, flux_tol=1e-6, max_iter=1000)
x_nodes, x_centers, _ = geom.build_mesh()
print(f"keff = {det.keff:.6f}")

# 打印界面附近的扩散通量
print(f"\n界面附近扩散通量 (x=10和x=50附近):")
print(f"{'x(cm)':>8} | {'phi':>12} | {'区域':>15}")
print(f"{'-'*45}")
for xi in [8.0, 9.0, 9.5, 9.9, 10.1, 10.5, 11.0, 12.0]:
    idx = np.argmin(np.abs(x_centers - xi))
    xc = x_centers[idx]
    mat = _get_material_at_x(geom, xc)
    print(f"{xc:>8.3f} | {det.phi[idx]:>12.6f} | {mat.name:>15}")

# 2. 蒙特卡洛模拟
print("\n--- 蒙特卡洛模拟 ---")
mc = run_monte_carlo_1d(
    geom,
    n_neutrons_per_gen=5000,
    n_generations=60,
    n_discard=20,
    n_flux_bins=120,  # 每cm 2个bin，和扩散方程差不多
    seed=42,
)
print(f"keff = {mc.keff_mean:.6f} ± {mc.keff_std:.6f}")
print(f"总碰撞次数: {mc.total_collisions:,}")

# 归一化：用中心区域(x=20-40)的平均通量来归一，这样中心区域应该一致
mc_center_mask = (mc.flux_centers >= 20.0) & (mc.flux_centers <= 40.0)
det_center_mask = (x_centers >= 20.0) & (x_centers <= 40.0)
mc_center_avg = np.mean(mc.flux_mc[mc_center_mask])
det_center_avg = np.mean(det.phi[det_center_mask])
mc_norm = mc.flux_mc / mc_center_avg
det_norm = det.phi / det_center_avg

print(f"\n中心区域平均通量: MC={mc_center_avg:.4f}, 扩散={det_center_avg:.4f}")
print(f"已归一化: 中心区域平均=1")

# 计算插值后的扩散通量（在MC的bin中心上）
det_at_mc = np.interp(mc.flux_centers, x_centers, det.phi, left=0.0, right=0.0)
det_at_mc_norm = det_at_mc / det_center_avg

# 详细打印界面附近的MC通量和扩散通量
print(f"\n=== 左界面 x=10cm 附近的详细对比 ===")
print(f"{'x(cm)':>8} | {'MC_norm':>10} | {'扩散_norm':>10} | {'相对误差%':>10} | {'区域':>15}")
print(f"{'-'*65}")

for xi in np.arange(2.0, 18.0, 1.0):
    idx_mc = np.argmin(np.abs(mc.flux_centers - xi))
    xc_mc = mc.flux_centers[idx_mc]
    mc_val = mc_norm[idx_mc]
    det_val = det_at_mc_norm[idx_mc]
    mat = _get_material_at_x(geom, xc_mc)
    
    if det_val > 1e-6:
        err = (mc_val - det_val) / det_val * 100
    else:
        err = float('inf')
    
    # 标记界面
    marker = " ◄── 界面" if abs(xc_mc - 10.0) < 1.0 else ""
    print(f"{xc_mc:>8.3f} | {mc_val:>10.4f} | {det_val:>10.4f} | {err:>9.2f}% | {mat.name:>15}{marker}")

print(f"\n=== 右界面 x=50cm 附近的详细对比 ===")
print(f"{'x(cm)':>8} | {'MC_norm':>10} | {'扩散_norm':>10} | {'相对误差%':>10} | {'区域':>15}")
print(f"{'-'*65}")

for xi in np.arange(42.0, 58.0, 1.0):
    idx_mc = np.argmin(np.abs(mc.flux_centers - xi))
    xc_mc = mc.flux_centers[idx_mc]
    mc_val = mc_norm[idx_mc]
    det_val = det_at_mc_norm[idx_mc]
    mat = _get_material_at_x(geom, xc_mc)
    
    if det_val > 1e-6:
        err = (mc_val - det_val) / det_val * 100
    else:
        err = float('inf')
    
    marker = " ◄── 界面" if abs(xc_mc - 50.0) < 1.0 else ""
    print(f"{xc_mc:>8.3f} | {mc_val:>10.4f} | {det_val:>10.4f} | {err:>9.2f}% | {mat.name:>15}{marker}")

# 按区域统计误差
print(f"\n=== 分区域误差统计 ===")
region_defs = [
    ("左反射层", 0.0, 10.0),
    ("燃料区", 10.0, 50.0),
    ("右反射层", 50.0, 60.0),
]

for name, x_start, x_end in region_defs:
    mask = (mc.flux_centers >= x_start) & (mc.flux_centers < x_end)
    mc_vals = mc_norm[mask]
    det_vals = det_at_mc_norm[mask]
    valid = det_vals > 1e-6
    if np.any(valid):
        errors = np.abs(mc_vals[valid] - det_vals[valid]) / det_vals[valid] * 100
        mean_err = np.mean(errors)
        max_err = np.max(errors)
        print(f"  {name}[{x_start},{x_end}]: 平均误差={mean_err:.2f}%, 最大误差={max_err:.2f}%")

print("\n" + "=" * 70)
print("诊断完成")
print("=" * 70)
