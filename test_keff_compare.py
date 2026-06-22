import numpy as np
import sys
sys.path.insert(0, '.')

from neutron_diffusion.geometry import Geometry1D, Region1D, BoundaryCondition
from neutron_diffusion.materials import get_preset_1g
from neutron_diffusion.eigenvalue import criticality_1d_1g
from neutron_diffusion.monte_carlo import run_monte_carlo_1d

print("=" * 70)
print("判定实验：单区域 vs 多区域 keff 差异")
print("=" * 70)

# 实验1：单区域均匀燃料，40cm宽，真空边界
print("\n--- 实验1：单区域均匀燃料 40cm (真空边界) ---")
geom1 = Geometry1D(
    regions=[Region1D(0.0, 40.0, "UO2燃料(3%)")],
    dx=0.5,
    bc=BoundaryCondition(left="vacuum", right="vacuum"),
    material_mode="1g"
)
det1 = criticality_1d_1g(geom1, k_tol=1e-7, flux_tol=1e-6, max_iter=1000)
mc1 = run_monte_carlo_1d(geom1, n_neutrons_per_gen=5000, n_generations=50, 
                         n_discard=20, n_flux_bins=80, seed=42)
diff1 = (det1.keff - mc1.keff_mean) / mc1.keff_mean * 1e5
print(f"  扩散 keff = {det1.keff:.6f}")
print(f"  MC   keff = {mc1.keff_mean:.6f} ± {mc1.keff_std:.6f}")
print(f"  差异      = {diff1:.0f} pcm")

# 实验2：单区域均匀燃料，80cm宽（更大，泄漏更少，扩散近似应该更好）
print("\n--- 实验2：单区域均匀燃料 80cm (真空边界) ---")
geom2 = Geometry1D(
    regions=[Region1D(0.0, 80.0, "UO2燃料(3%)")],
    dx=0.5,
    bc=BoundaryCondition(left="vacuum", right="vacuum"),
    material_mode="1g"
)
det2 = criticality_1d_1g(geom2, k_tol=1e-7, flux_tol=1e-6, max_iter=1000)
mc2 = run_monte_carlo_1d(geom2, n_neutrons_per_gen=5000, n_generations=50,
                         n_discard=20, n_flux_bins=160, seed=42)
diff2 = (det2.keff - mc2.keff_mean) / mc2.keff_mean * 1e5
print(f"  扩散 keff = {det2.keff:.6f}")
print(f"  MC   keff = {mc2.keff_mean:.6f} ± {mc2.keff_std:.6f}")
print(f"  差异      = {diff2:.0f} pcm")

# 实验3：反射层+燃料+反射层，总宽60cm（燃料40+反射层各10），真空边界
print("\n--- 实验3：反射层(10)+燃料(40)+反射层(10) 总宽60cm (真空边界) ---")
regions3 = [
    Region1D(0.0, 10.0, "石墨反射层"),
    Region1D(10.0, 50.0, "UO2燃料(3%)"),
    Region1D(50.0, 60.0, "石墨反射层"),
]
geom3 = Geometry1D(
    regions=regions3,
    dx=0.5,
    bc=BoundaryCondition(left="vacuum", right="vacuum"),
    material_mode="1g"
)
det3 = criticality_1d_1g(geom3, k_tol=1e-7, flux_tol=1e-6, max_iter=1000)
mc3 = run_monte_carlo_1d(geom3, n_neutrons_per_gen=5000, n_generations=50,
                         n_discard=20, n_flux_bins=120, seed=42)
diff3 = (det3.keff - mc3.keff_mean) / mc3.keff_mean * 1e5
print(f"  扩散 keff = {det3.keff:.6f}")
print(f"  MC   keff = {mc3.keff_mean:.6f} ± {mc3.keff_std:.6f}")
print(f"  差异      = {diff3:.0f} pcm")

# 实验4：反射层+燃料+反射层，但用零通量边界（φ=0在真实边界）
print("\n--- 实验4：反射层+燃料+反射层 (零通量边界) ---")
geom4 = Geometry1D(
    regions=regions3,
    dx=0.5,
    bc=BoundaryCondition(left="zero_flux", right="zero_flux"),
    material_mode="1g"
)
det4 = criticality_1d_1g(geom4, k_tol=1e-7, flux_tol=1e-6, max_iter=1000)
mc4 = run_monte_carlo_1d(geom4, n_neutrons_per_gen=5000, n_generations=50,
                         n_discard=20, n_flux_bins=120, seed=42)
diff4 = (det4.keff - mc4.keff_mean) / mc4.keff_mean * 1e5
print(f"  扩散 keff = {det4.keff:.6f}")
print(f"  MC   keff = {mc4.keff_mean:.6f} ± {mc4.keff_std:.6f}")
print(f"  差异      = {diff4:.0f} pcm")

# 总结
print("\n" + "=" * 70)
print("总结：")
print(f"  单区域40cm燃料(真空):   {diff1:.0f} pcm")
print(f"  单区域80cm燃料(真空):   {diff2:.0f} pcm")
print(f"  三区域带反射层(真空):   {diff3:.0f} pcm")
print(f"  三区域带反射层(零通量): {diff4:.0f} pcm")
print("=" * 70)
