import streamlit as st
import numpy as np
from copy import deepcopy

from neutron_diffusion.geometry import (
    Geometry1D, Geometry2D, Region1D, Region2D,
    BoundaryCondition
)
from neutron_diffusion.materials import (
    PRESET_MATERIALS_1G, PRESET_MATERIALS_2G,
    Material1G, Material2G, get_preset_1g, get_preset_2g,
    get_all_material_names_1g, get_all_material_names_2g,
    register_custom_1g, register_custom_2g, clear_custom_materials
)
from neutron_diffusion.eigenvalue import (
    criticality_1d_1g, criticality_2d_1g,
    criticality_1d_2g, criticality_2d_2g,
    EigenvalueResult
)
from neutron_diffusion.solver_1g import solve_1d_diffusion_1g, solve_2d_diffusion_1g_sor
from neutron_diffusion.solver_2g import solve_1d_diffusion_2g, solve_2d_diffusion_2g_sor
from neutron_diffusion.control_rod import (
    control_rod_1d, control_rod_2d,
    control_rod_1d_step_insertion, ControlRodResult
)
from neutron_diffusion.sensitivity import (
    reflector_thickness_sensitivity_1d,
    find_critical_size_1d
)
from neutron_diffusion.benchmark import (
    run_iaea_benchmark, IAEA_2G_LWR_KEFF_REF,
    BenchmarkResult
)
from neutron_diffusion.visualization import (
    plot_1d_flux, plot_2d_flux, plot_2d_contour,
    plot_convergence, plot_power_distribution_1d,
    plot_power_distribution_2d, plot_sensitivity,
    plot_control_rod_worth, plot_1d_profile,
    plot_kinetics_power, plot_kinetics_reactivity,
    plot_kinetics_precursors, plot_kinetics_all,
    plot_keff_convergence_mc, plot_flux_comparison,
    plot_relative_error, plot_collision_histogram
)
from neutron_diffusion.monte_carlo import (
    run_monte_carlo_1d, compute_flux_relative_error, MonteCarloResult
)
from neutron_diffusion.kinetics import (
    DelayedNeutronParams, KineticsResult,
    get_u235_params, get_u238_params, get_pu239_params,
    reactivity_step, reactivity_linear, reactivity_sinusoidal,
    solve_point_kinetics_rk4, solve_point_kinetics_implicit_euler
)

st.set_page_config(page_title="核反应堆中子扩散方程求解器", layout="wide")
st.title("🔬 核反应堆中子扩散方程数值求解与临界计算工具")
st.sidebar.title("⚙️ 计算设置")

page = st.sidebar.radio(
    "选择功能模块",
    [
        "一维反应堆临界计算",
        "二维反应堆临界计算",
        "控制棒价值计算",
        "参数敏感性分析",
        "IAEA基准验证",
        "瞬态动力学模拟",
        "蒙特卡洛模拟"
    ]
)

BOUNDARY_OPTIONS = ["vacuum", "reflective", "zero_flux"]

with st.sidebar.expander("🧪 自定义材料参数", expanded=False):
    st.markdown("#### 单群自定义材料")
    n_custom_1g = st.number_input("单群自定义材料数量", 0, 10, 0, 1, key="n_custom_1g")
    for ci in range(n_custom_1g):
        st.markdown(f"**自定义材料 1G-{ci+1}**")
        cname = st.text_input(f"名称", f"自定义{ci+1}", key=f"cname_1g_{ci}")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            cSigma_a = st.number_input("Σ_a (cm⁻¹)", 0.0, 10.0, 0.1, 0.001, key=f"cSa_1g_{ci}", format="%.4f")
            cSigma_f = st.number_input("Σ_f (cm⁻¹)", 0.0, 10.0, 0.05, 0.001, key=f"cSf_1g_{ci}", format="%.4f")
        with cc2:
            cnu = st.number_input("ν", 0.0, 5.0, 2.42, 0.01, key=f"cnu_1g_{ci}", format="%.3f")
            cD = st.number_input("D (cm)", 0.001, 100.0, 1.2, 0.01, key=f"cD_1g_{ci}", format="%.3f")
        with cc3:
            cSigma_s = st.number_input("Σ_s (cm⁻¹)", 0.0, 10.0, 0.4, 0.01, key=f"cSs_1g_{ci}", format="%.4f")
        custom_mat_1g = Material1G(
            name=cname, Sigma_a=cSigma_a, Sigma_f=cSigma_f,
            nu=cnu, D=cD, Sigma_s=cSigma_s
        )
        register_custom_1g(cname, custom_mat_1g)

    st.divider()
    st.markdown("#### 两群自定义材料")
    n_custom_2g = st.number_input("两群自定义材料数量", 0, 10, 0, 1, key="n_custom_2g")
    for ci in range(n_custom_2g):
        st.markdown(f"**自定义材料 2G-{ci+1}**")
        cname2 = st.text_input(f"名称", f"自定义{ci+1}", key=f"cname_2g_{ci}")
        st.markdown("**快群 (群1)**")
        cc1a, cc1b = st.columns(2)
        with cc1a:
            cD1 = st.number_input("D₁ (cm)", 0.001, 100.0, 1.5, 0.01, key=f"cD1_2g_{ci}", format="%.3f")
            cSigma_a1 = st.number_input("Σ_a₁ (cm⁻¹)", 0.0, 10.0, 0.01, 0.001, key=f"cSa1_2g_{ci}", format="%.4f")
        with cc1b:
            cSigma_f1 = st.number_input("Σ_f₁ (cm⁻¹)", 0.0, 10.0, 0.005, 0.001, key=f"cSf1_2g_{ci}", format="%.4f")
            cnu1 = st.number_input("ν₁", 0.0, 5.0, 2.42, 0.01, key=f"cnu1_2g_{ci}", format="%.3f")
        st.markdown("**热群 (群2)**")
        cc2a, cc2b, cc2c = st.columns(3)
        with cc2a:
            cD2 = st.number_input("D₂ (cm)", 0.001, 100.0, 0.8, 0.01, key=f"cD2_2g_{ci}", format="%.3f")
            cSigma_a2 = st.number_input("Σ_a₂ (cm⁻¹)", 0.0, 10.0, 0.12, 0.001, key=f"cSa2_2g_{ci}", format="%.4f")
        with cc2b:
            cSigma_f2 = st.number_input("Σ_f₂ (cm⁻¹)", 0.0, 10.0, 0.08, 0.001, key=f"cSf2_2g_{ci}", format="%.4f")
            cnu2 = st.number_input("ν₂", 0.0, 5.0, 2.42, 0.01, key=f"cnu2_2g_{ci}", format="%.3f")
        with cc2c:
            cSigma_s12 = st.number_input("Σ_s₁₂ (cm⁻¹)", 0.0, 10.0, 0.08, 0.001, key=f"cSs12_2g_{ci}", format="%.4f")
        custom_mat_2g = Material2G(
            name=cname2, D1=cD1, Sigma_a1=cSigma_a1, Sigma_f1=cSigma_f1, nu1=cnu1,
            D2=cD2, Sigma_a2=cSigma_a2, Sigma_f2=cSigma_f2, nu2=cnu2,
            Sigma_s12=cSigma_s12
        )
        register_custom_2g(cname2, custom_mat_2g)

MATERIAL_OPTIONS = get_all_material_names_1g()
MATERIAL_OPTIONS_2G = get_all_material_names_2g()


def display_result_summary(result: EigenvalueResult):
    col1, col2, col3 = st.columns(3)
    col1.metric("临界特征值 keff", f"{result.keff:.6f}")
    col2.metric("是否收敛", "✅ 是" if result.converged else "❌ 否")
    col3.metric("外迭代次数", result.n_iterations)

    if result.keff > 1.0:
        st.success(f"⚡ 反应堆超临界 (keff = {result.keff:.6f} > 1.0)")
    elif result.keff < 1.0:
        st.warning(f"📉 反应堆次临界 (keff = {result.keff:.6f} < 1.0)")
    else:
        st.info(f"⚖️ 反应堆恰好临界 (keff = {result.keff:.6f} = 1.0)")


if page == "一维反应堆临界计算":
    st.header("📐 一维平板反应堆临界计算")

    with st.sidebar:
        st.subheader("能量群设置")
        ngroups_1d = st.radio("能群数", ["单群", "两群"], horizontal=True)
        ngroups_1d_key = "1g" if ngroups_1d == "单群" else "2g"

        st.subheader("网格设置")
        dx_1d = st.slider("网格间距 (cm)", 0.1, 5.0, 1.0, 0.1)

        st.subheader("边界条件")
        bc_left = st.selectbox("左边界", BOUNDARY_OPTIONS, index=0)
        bc_right = st.selectbox("右边界", BOUNDARY_OPTIONS, index=0)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🏗️ 几何与材料定义")
        n_regions = st.number_input("材料区域数", 1, 10, 2, 1, key="n_regions_1d")
        regions_1d = []
        current_x = 0.0

        for i in range(n_regions):
            st.markdown(f"**区域 {i+1}**")
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                width = st.number_input(f"宽度 (cm) - 区域{i+1}", 1.0, 200.0, 20.0, 1.0, key=f"w1d_{i}")
            with rcol2:
                mat_name = st.selectbox(f"材料 - 区域{i+1}", MATERIAL_OPTIONS,
                                       index=i % len(MATERIAL_OPTIONS), key=f"m1d_{i}")
            x_start = current_x
            x_end = current_x + width
            regions_1d.append(Region1D(x_start, x_end, mat_name))
            current_x = x_end
            st.info(f"x ∈ [{x_start:.1f}, {x_end:.1f}] cm: {mat_name}")

    with col2:
        st.subheader("🧮 求解器参数")
        k_tol = st.number_input("keff收敛阈值", 1e-7, 1e-3, 1e-5, format="%.1e")
        flux_tol = st.number_input("通量收敛阈值", 1e-6, 1e-2, 1e-4, format="%.1e")
        max_iter = st.number_input("最大外迭代次数", 50, 2000, 500, 50)

        if st.button("🚀 开始临界计算", type="primary", use_container_width=True):
            bc = BoundaryCondition(left=bc_left, right=bc_right)
            geom = Geometry1D(regions=regions_1d, dx=dx_1d, bc=bc, material_mode=ngroups_1d_key)

            with st.spinner("正在计算..."):
                if ngroups_1d_key == "1g":
                    result = criticality_1d_1g(geom, k_tol=k_tol, flux_tol=flux_tol, max_iter=max_iter)
                else:
                    result = criticality_1d_2g(geom, k_tol=k_tol, flux_tol=flux_tol, max_iter=max_iter)

            st.session_state["result_1d"] = result
            st.session_state["geom_1d"] = geom
            st.session_state["ngroups_1d"] = ngroups_1d_key

    if "result_1d" in st.session_state:
        result = st.session_state["result_1d"]
        geom = st.session_state["geom_1d"]
        ng_key = st.session_state["ngroups_1d"]

        st.divider()
        display_result_summary(result)

        x_nodes, x_centers, _ = geom.build_mesh()

        tab1, tab2, tab3 = st.tabs(["📊 通量分布", "⚡ 功率分布", "📈 收敛曲线"])

        with tab1:
            if ng_key == "1g":
                fig = plot_1d_flux(x_centers, result.phi, title="一维单群中子通量分布")
            else:
                fig = plot_1d_flux(x_centers, result.phi2, phi2=result.phi1,
                                  title="一维两群中子通量分布")
            st.pyplot(fig)

        with tab2:
            materials = geom.get_materials_1g() if ng_key == "1g" else geom.get_materials_2g()
            if ng_key == "1g":
                Sigma_f = np.array([m.Sigma_f for m in materials])
                power = Sigma_f * result.phi
            else:
                Sigma_f1 = np.array([m.Sigma_f1 for m in materials])
                Sigma_f2 = np.array([m.Sigma_f2 for m in materials])
                power = Sigma_f1 * result.phi1 + Sigma_f2 * result.phi2
            power = power / np.mean(power) if np.mean(power) > 0 else power
            peak_factor = np.max(power) / np.mean(power) if np.mean(power) > 0 else 1.0
            pcol1, pcol2, pcol3 = st.columns(3)
            pcol1.metric("功率峰因子", f"{peak_factor:.4f}")
            pcol2.metric("最大功率", f"{np.max(power):.4f}")
            pcol3.metric("平均功率", f"{np.mean(power):.4f}")
            fig_p = plot_power_distribution_1d(x_centers, power)
            st.pyplot(fig_p)

        with tab3:
            fig_c = plot_convergence(result.keff_history)
            st.pyplot(fig_c)
            if result.flux_changes:
                fig_f = plot_1d_profile(
                    np.arange(len(result.flux_changes)),
                    result.flux_changes,
                    title="通量最大相对变化",
                    xlabel="外迭代次数"
                )
                st.pyplot(fig_f)

elif page == "二维反应堆临界计算":
    st.header("📐 二维矩形反应堆临界计算")

    with st.sidebar:
        st.subheader("能量群设置")
        ngroups_2d = st.radio("能群数", ["单群", "两群"], horizontal=True, key="ng2d")
        ngroups_2d_key = "1g" if ngroups_2d == "单群" else "2g"

        st.subheader("网格设置")
        dx_2d = st.slider("X方向网格间距 (cm)", 0.1, 10.0, 2.0, 0.5, key="dx2d")
        dy_2d = st.slider("Y方向网格间距 (cm)", 0.1, 10.0, 2.0, 0.5, key="dy2d")

        st.subheader("边界条件")
        bc_left_2d = st.selectbox("左边界", BOUNDARY_OPTIONS, index=0, key="bc_l2d")
        bc_right_2d = st.selectbox("右边界", BOUNDARY_OPTIONS, index=0, key="bc_r2d")
        bc_bottom_2d = st.selectbox("下边界", BOUNDARY_OPTIONS, index=0, key="bc_b2d")
        bc_top_2d = st.selectbox("上边界", BOUNDARY_OPTIONS, index=0, key="bc_t2d")

        st.subheader("SOR求解器")
        omega = st.slider("超松弛因子 ω (1.0=Gauss-Seidel, 1.2~1.4=推荐SOR)", 1.0, 1.6, 1.2, 0.05)
        inner_tol = st.number_input("内迭代收敛阈值", 1e-8, 1e-3, 1e-6, format="%.1e", key="it2d")
        inner_max_iter = st.number_input("最大内迭代次数", 100, 10000, 2000, 100, key="imi2d")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🏗️ 几何与材料定义")
        total_width = st.number_input("堆芯X方向总宽度 (cm)", 20.0, 500.0, 100.0, 5.0, key="tw2d")
        total_height = st.number_input("堆芯Y方向总高度 (cm)", 20.0, 500.0, 100.0, 5.0, key="th2d")
        n_regions_2d = st.number_input("矩形区域数", 1, 20, 2, 1, key="nr2d")
        regions_2d = []

        for i in range(n_regions_2d):
            st.markdown(f"**区域 {i+1}**")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                xs = st.number_input("x起始", 0.0, total_width, 0.0 if i == 0 else total_width/2, 1.0, key=f"xs_{i}")
            with c2:
                xe = st.number_input("x结束", 0.0, total_width, total_width/2 if i == 0 else total_width, 1.0, key=f"xe_{i}")
            with c3:
                ys = st.number_input("y起始", 0.0, total_height, 0.0 if i < 2 else total_height/2, 1.0, key=f"ys_{i}")
            with c4:
                ye = st.number_input("y结束", 0.0, total_height, total_height/2 if i < 2 else total_height, 1.0, key=f"ye_{i}")
            with c5:
                mat = st.selectbox("材料", MATERIAL_OPTIONS, index=i % len(MATERIAL_OPTIONS), key=f"m2d_{i}")
            regions_2d.append(Region2D(xs, xe, ys, ye, mat))

    with col2:
        st.subheader("🧮 求解器参数")
        k_tol_2d = st.number_input("keff收敛阈值", 1e-7, 1e-3, 1e-5, format="%.1e", key="kt2d")
        flux_tol_2d = st.number_input("通量收敛阈值", 1e-6, 1e-2, 1e-4, format="%.1e", key="ft2d")
        max_iter_2d = st.number_input("最大外迭代次数", 50, 2000, 500, 50, key="mi2d")

        if st.button("🚀 开始临界计算", type="primary", use_container_width=True, key="run2d"):
            bc = BoundaryCondition(
                left=bc_left_2d, right=bc_right_2d,
                bottom=bc_bottom_2d, top=bc_top_2d
            )
            geom = Geometry2D(
                regions=regions_2d, dx=dx_2d, dy=dy_2d,
                bc=bc, material_mode=ngroups_2d_key
            )

            with st.spinner("正在计算二维问题（可能需要一些时间）..."):
                if ngroups_2d_key == "1g":
                    result = criticality_2d_1g(
                        geom, omega=omega, inner_tol=inner_tol,
                        inner_max_iter=inner_max_iter,
                        k_tol=k_tol_2d, flux_tol=flux_tol_2d,
                        max_iter=max_iter_2d
                    )
                else:
                    result = criticality_2d_2g(
                        geom, omega=omega, inner_tol=inner_tol,
                        inner_max_iter=inner_max_iter,
                        k_tol=k_tol_2d, flux_tol=flux_tol_2d,
                        max_iter=max_iter_2d
                    )

            st.session_state["result_2d"] = result
            st.session_state["geom_2d"] = geom
            st.session_state["ngroups_2d"] = ngroups_2d_key

    if "result_2d" in st.session_state:
        result = st.session_state["result_2d"]
        geom = st.session_state["geom_2d"]
        ng_key = st.session_state["ngroups_2d"]

        st.divider()
        display_result_summary(result)
        if result.inner_iterations:
            st.caption(f"平均内迭代次数: {np.mean(result.inner_iterations):.0f}")

        _, _, x_centers, y_centers, _ = geom.build_mesh()

        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ 通量热力图", "📊 通量等值线", "⚡ 功率分布", "📈 收敛曲线"])

        with tab1:
            if ng_key == "1g":
                fig = plot_2d_flux(x_centers, y_centers, result.phi, title="二维单群中子通量热力图")
            else:
                st.markdown("**快群通量 φ₁**")
                fig1 = plot_2d_flux(x_centers, y_centers, result.phi1,
                                   title="快群通量 φ₁ 热力图", cmap="plasma")
                st.pyplot(fig1)
                st.markdown("**热群通量 φ₂**")
                fig2 = plot_2d_flux(x_centers, y_centers, result.phi2,
                                   title="热群通量 φ₂ 热力图", cmap="viridis")
                st.pyplot(fig2)
            if ng_key == "1g":
                st.pyplot(fig)

            st.subheader("一维剖面截取")
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                profile_type = st.radio("剖面方向", ["水平", "垂直"])
            with pcol2:
                if profile_type == "水平":
                    y_idx = st.slider("Y索引", 0, len(y_centers)-1, len(y_centers)//2)
                    if ng_key == "1g":
                        fig_p = plot_1d_profile(x_centers, result.phi[y_idx, :],
                                               title=f"Y = {y_centers[y_idx]:.1f} cm 处水平通量剖面")
                    else:
                        st.markdown("快群剖面")
                        fig_p1 = plot_1d_profile(x_centers, result.phi1[y_idx, :],
                                                title=f"Y = {y_centers[y_idx]:.1f} cm 处快群水平剖面")
                        st.pyplot(fig_p1)
                        fig_p = plot_1d_profile(x_centers, result.phi2[y_idx, :],
                                               title=f"Y = {y_centers[y_idx]:.1f} cm 处热群水平剖面")
                else:
                    x_idx = st.slider("X索引", 0, len(x_centers)-1, len(x_centers)//2)
                    if ng_key == "1g":
                        fig_p = plot_1d_profile(y_centers, result.phi[:, x_idx],
                                               title=f"X = {x_centers[x_idx]:.1f} cm 处垂直通量剖面",
                                               xlabel="y (cm)")
                    else:
                        st.markdown("快群剖面")
                        fig_p1 = plot_1d_profile(y_centers, result.phi1[:, x_idx],
                                                title=f"X = {x_centers[x_idx]:.1f} cm 处快群垂直剖面",
                                                xlabel="y (cm)")
                        st.pyplot(fig_p1)
                        fig_p = plot_1d_profile(y_centers, result.phi2[:, x_idx],
                                               title=f"X = {x_centers[x_idx]:.1f} cm 处热群垂直剖面",
                                               xlabel="y (cm)")
            st.pyplot(fig_p)

        with tab2:
            if ng_key == "1g":
                fig = plot_2d_contour(x_centers, y_centers, result.phi, title="二维单群通量等值线图")
                st.pyplot(fig)
            else:
                st.markdown("**快群通量 φ₁ 等值线**")
                fig1 = plot_2d_contour(x_centers, y_centers, result.phi1,
                                      title="快群通量 φ₁ 等值线图", cmap="plasma")
                st.pyplot(fig1)
                st.markdown("**热群通量 φ₂ 等值线**")
                fig2 = plot_2d_contour(x_centers, y_centers, result.phi2,
                                      title="热群通量 φ₂ 等值线图", cmap="viridis")
                st.pyplot(fig2)

        with tab3:
            materials_2d = geom.get_materials_1g() if ng_key == "1g" else geom.get_materials_2g()
            if ng_key == "1g":
                Sigma_f = np.array([[m.Sigma_f for m in row] for row in materials_2d])
                power = Sigma_f * result.phi
            else:
                Sigma_f1 = np.array([[m.Sigma_f1 for m in row] for row in materials_2d])
                Sigma_f2 = np.array([[m.Sigma_f2 for m in row] for row in materials_2d])
                power = Sigma_f1 * result.phi1 + Sigma_f2 * result.phi2
            power = power / np.mean(power) if np.mean(power) > 0 else power
            peak_factor_2d = np.max(power) / np.mean(power) if np.mean(power) > 0 else 1.0
            pcol1, pcol2, pcol3 = st.columns(3)
            pcol1.metric("功率峰因子", f"{peak_factor_2d:.4f}")
            pcol2.metric("最大功率", f"{np.max(power):.4f}")
            pcol3.metric("平均功率", f"{np.mean(power):.4f}")
            fig_p = plot_power_distribution_2d(x_centers, y_centers, power)
            st.pyplot(fig_p)

        with tab4:
            fig_c = plot_convergence(result.keff_history)
            st.pyplot(fig_c)

elif page == "控制棒价值计算":
    st.header("🎛️ 控制棒价值计算")

    calc_mode = st.radio("计算模式", ["一维控制棒", "二维控制棒", "一维逐步插入"], horizontal=True)

    if calc_mode == "一维控制棒" or calc_mode == "一维逐步插入":
        with st.sidebar:
            ngroups_cr = st.radio("能群数", ["单群", "两群"], horizontal=True, key="cr_ng")
            ngroups_cr_key = "1g" if ngroups_cr == "单群" else "2g"
            dx_cr = st.slider("网格间距 (cm)", 0.1, 5.0, 0.5, 0.1, key="cr_dx")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏗️ 堆芯几何")
            n_regions_cr = st.number_input("材料区域数", 1, 10, 3, 1, key="cr_nr")
            regions_cr = []
            current_x_cr = 0.0
            for i in range(n_regions_cr):
                rcol1, rcol2 = st.columns(2)
                with rcol1:
                    w = st.number_input(f"宽度区域{i+1} (cm)", 1.0, 200.0,
                                       10.0 if i == 0 or i == 2 else 30.0, 1.0, key=f"cr_w{i}")
                with rcol2:
                    m = st.selectbox(f"材料区域{i+1}", MATERIAL_OPTIONS,
                                    index=2 if i == 0 or i == 2 else 0, key=f"cr_m{i}")
                regions_cr.append(Region1D(current_x_cr, current_x_cr + w, m))
                current_x_cr += w

        with col2:
            st.subheader("🎚️ 控制棒位置")
            rod_x_start = st.number_input("控制棒起始位置 (cm)", 0.0, current_x_cr, 20.0, 1.0)
            rod_x_end = st.number_input("控制棒结束位置 (cm)", 0.0, current_x_cr, 30.0, 1.0)

            if calc_mode == "一维逐步插入":
                n_steps = st.number_input("插入步数", 2, 50, 10, 1)
                if st.button("🚀 逐步插入计算", type="primary", use_container_width=True, key="run_cr_step"):
                    bc = BoundaryCondition(left="vacuum", right="vacuum")
                    geom = Geometry1D(regions=regions_cr, dx=dx_cr, bc=bc, material_mode=ngroups_cr_key)
                    with st.spinner("正在计算控制棒逐步插入..."):
                        cr_result = control_rod_1d_step_insertion(
                            geom, rod_x_start, rod_x_end, n_steps, ngroups_cr_key
                        )
                    st.session_state["cr_result"] = cr_result
            else:
                if st.button("🚀 计算控制棒价值", type="primary", use_container_width=True, key="run_cr_1d"):
                    bc = BoundaryCondition(left="vacuum", right="vacuum")
                    geom = Geometry1D(regions=regions_cr, dx=dx_cr, bc=bc, material_mode=ngroups_cr_key)
                    with st.spinner("正在计算..."):
                        cr_result = control_rod_1d(geom, (rod_x_start, rod_x_end), ngroups_cr_key)
                    st.session_state["cr_result"] = cr_result

        if "cr_result" in st.session_state:
            cr_result = st.session_state["cr_result"]
            st.divider()
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("无棒 keff_out", f"{cr_result.keff_out:.6f}")
            col_b.metric("插棒 keff_in", f"{cr_result.keff_in:.6f}")
            col_c.metric("控制棒价值", f"{cr_result.reactivity_pcm:.1f} pcm")

            if cr_result.step_results:
                fig = plot_control_rod_worth(cr_result.step_results)
                st.pyplot(fig)

    elif calc_mode == "二维控制棒":
        with st.sidebar:
            ngroups_cr2d = st.radio("能群数", ["单群", "两群"], horizontal=True, key="cr2d_ng")
            ngroups_cr2d_key = "1g" if ngroups_cr2d == "单群" else "2g"
            dx_cr2d = st.slider("X方向网格间距 (cm)", 0.5, 10.0, 2.0, 0.5, key="cr2d_dx")
            dy_cr2d = st.slider("Y方向网格间距 (cm)", 0.5, 10.0, 2.0, 0.5, key="cr2d_dy")
            omega_cr = st.slider("SOR因子 ω (1.0=Gauss-Seidel, 1.2~1.4=SOR)", 1.0, 1.6, 1.2, 0.05, key="cr2d_omega")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏗️ 堆芯几何")
            total_w = st.number_input("总宽度 (cm)", 50.0, 300.0, 100.0, 5.0, key="cr2d_tw")
            total_h = st.number_input("总高度 (cm)", 50.0, 300.0, 100.0, 5.0, key="cr2d_th")
            regions_cr2d = [
                Region2D(0.0, total_w, 0.0, total_h, "UO2燃料(3%)")
            ]

        with col2:
            st.subheader("🎚️ 控制棒区域")
            n_rods = st.number_input("控制棒区域数", 1, 10, 1, 1, key="cr2d_nr")
            rod_list = []
            for i in range(n_rods):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    rxs = st.number_input("x起始", 0.0, total_w, 45.0 + i*10, 1.0, key=f"rods_xs{i}")
                with c2:
                    rxe = st.number_input("x结束", 0.0, total_w, 55.0 + i*10, 1.0, key=f"rods_xe{i}")
                with c3:
                    rys = st.number_input("y起始", 0.0, total_h, 45.0, 1.0, key=f"rods_ys{i}")
                with c4:
                    rye = st.number_input("y结束", 0.0, total_h, 55.0, 1.0, key=f"rods_ye{i}")
                rod_list.append((rxs, rxe, rys, rye))

            if st.button("🚀 计算控制棒价值", type="primary", use_container_width=True, key="run_cr_2d"):
                bc = BoundaryCondition(left="vacuum", right="vacuum", bottom="vacuum", top="vacuum")
                geom = Geometry2D(
                    regions=regions_cr2d, dx=dx_cr2d, dy=dy_cr2d,
                    bc=bc, material_mode=ngroups_cr2d_key
                )
                with st.spinner("正在计算二维控制棒价值..."):
                    cr_result = control_rod_2d(geom, rod_list, ngroups_cr2d_key, omega=omega_cr)
                st.session_state["cr_result"] = cr_result

        if "cr_result" in st.session_state:
            cr_result = st.session_state["cr_result"]
            st.divider()
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("无棒 keff_out", f"{cr_result.keff_out:.6f}")
            col_b.metric("插棒 keff_in", f"{cr_result.keff_in:.6f}")
            col_c.metric("控制棒价值", f"{cr_result.reactivity_pcm:.1f} pcm")

elif page == "参数敏感性分析":
    st.header("📈 参数敏感性分析与临界尺寸搜索")

    analysis_mode = st.radio("分析模式",
                            ["反射层厚度敏感性", "临界尺寸搜索（二分法）"],
                            horizontal=True)

    with st.sidebar:
        ngroups_sens = st.radio("能群数", ["单群", "两群"], horizontal=True, key="sens_ng")
        ngroups_sens_key = "1g" if ngroups_sens == "单群" else "2g"
        fuel_mat = st.selectbox("燃料材料", MATERIAL_OPTIONS, index=0, key="sens_fm")
        ref_mat = st.selectbox("反射层材料", MATERIAL_OPTIONS, index=2, key="sens_rm")
        dx_sens = st.slider("网格间距 (cm)", 0.1, 5.0, 0.5, 0.1, key="sens_dx")

    if analysis_mode == "反射层厚度敏感性":
        col1, col2 = st.columns(2)
        with col1:
            fuel_width = st.number_input("燃料区宽度 (cm)", 10.0, 200.0, 40.0, 1.0)
            n_points = st.number_input("采样点数", 3, 30, 10, 1)
            rt_min = st.number_input("反射层最小厚度 (cm)", 0.0, 50.0, 1.0, 0.5)
            rt_max = st.number_input("反射层最大厚度 (cm)", 1.0, 100.0, 30.0, 0.5)

        with col2:
            st.info(f"将在 [{rt_min}, {rt_max}] cm 范围内取 {n_points} 个点进行扫描")
            if st.button("🚀 开始敏感性分析", type="primary", use_container_width=True):
                rt_values = list(np.linspace(rt_min, rt_max, n_points))
                with st.spinner("正在计算参数敏感性..."):
                    x_vals, k_vals = reflector_thickness_sensitivity_1d(
                        fuel_width, rt_values, fuel_mat, ref_mat, dx_sens, ngroups_sens_key
                    )
                st.session_state["sens_data"] = (x_vals, k_vals)
                st.session_state["sens_param"] = "反射层厚度 (cm)"
                st.session_state["sens_title"] = "反射层厚度对keff的影响"

        if "sens_data" in st.session_state:
            x_vals, k_vals = st.session_state["sens_data"]
            fig = plot_sensitivity(x_vals, k_vals,
                                  param_name=st.session_state["sens_param"],
                                  title=st.session_state["sens_title"])
            st.pyplot(fig)

            data = {"反射层厚度 (cm)": x_vals, "keff": k_vals}
            st.dataframe(data, use_container_width=True)

    elif analysis_mode == "临界尺寸搜索（二分法）":
        col1, col2 = st.columns(2)
        with col1:
            rt = st.number_input("固定反射层厚度 (cm)", 0.0, 100.0, 10.0, 0.5)
            size_min = st.number_input("燃料区最小搜索尺寸 (cm)", 1.0, 100.0, 10.0, 1.0)
            size_max = st.number_input("燃料区最大搜索尺寸 (cm)", 10.0, 500.0, 100.0, 1.0)
            search_tol = st.number_input("搜索收敛容差", 1e-5, 1e-2, 1e-4, format="%.1e")

        with col2:
            st.info(f"将在 [{size_min}, {size_max}] cm 范围内用二分法搜索使 keff≈1.0 的燃料区尺寸")
            if st.button("🚀 开始临界尺寸搜索", type="primary", use_container_width=True):
                with st.spinner("正在用二分法搜索临界尺寸..."):
                    crit_size, sizes_hist, keffs_hist = find_critical_size_1d(
                        rt, size_min, size_max, fuel_mat, ref_mat,
                        dx_sens, ngroups_sens_key, tol=search_tol
                    )
                st.session_state["crit_data"] = (crit_size, sizes_hist, keffs_hist)

        if "crit_data" in st.session_state:
            crit_size, sizes_hist, keffs_hist = st.session_state["crit_data"]
            st.success(f"🎯 找到临界燃料区尺寸: **{crit_size:.3f} cm**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("临界燃料区尺寸", f"{crit_size:.3f} cm")
            with col_b:
                st.metric("最终 keff", f"{keffs_hist[-1]:.6f}")

            fig = plot_sensitivity(sizes_hist, keffs_hist,
                                  param_name="燃料区尺寸 (cm)",
                                  title="临界尺寸搜索过程 (二分法)")
            st.pyplot(fig)

elif page == "IAEA基准验证":
    st.header("✅ IAEA二维两群LWR基准验证")

    st.markdown(f"""
    **基准问题描述:**
    - 1/4堆芯对称模型，尺寸 170cm × 170cm
    - 左、下边界为反射边界，右、上边界为真空边界
    - 包含两种燃料区和反射层
    - 参考 keff = **{IAEA_2G_LWR_KEFF_REF}**
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚙️ 计算参数")
        dx_bm = st.slider("X方向网格间距 (cm)", 0.5, 10.0, 5.0, 0.5, key="bm_dx")
        dy_bm = st.slider("Y方向网格间距 (cm)", 0.5, 10.0, 5.0, 0.5, key="bm_dy")
        omega_bm = st.slider("SOR超松弛因子 ω (1.0=Gauss-Seidel, 1.2~1.4=SOR)", 1.0, 1.6, 1.2, 0.05, key="bm_omega")
        inner_tol_bm = st.number_input("内迭代收敛阈值", 1e-8, 1e-3, 1e-6, format="%.1e", key="bm_it")
        inner_max_bm = st.number_input("最大内迭代", 100, 10000, 2000, 100, key="bm_imi")
        k_tol_bm = st.number_input("keff收敛阈值", 1e-7, 1e-3, 1e-5, format="%.1e", key="bm_kt")
        flux_tol_bm = st.number_input("通量收敛阈值", 1e-6, 1e-2, 1e-4, format="%.1e", key="bm_ft")
        max_iter_bm = st.number_input("最大外迭代", 50, 2000, 500, 50, key="bm_mi")

    with col2:
        st.info("点击下方按钮运行IAEA基准题验证")
        if st.button("🚀 运行IAEA基准验证", type="primary", use_container_width=True, key="run_bm"):
            with st.spinner("正在运行IAEA二维两群基准题..."):
                bm_result = run_iaea_benchmark(
                    dx=dx_bm, dy=dy_bm, omega=omega_bm,
                    inner_tol=inner_tol_bm, inner_max_iter=inner_max_bm,
                    k_tol=k_tol_bm, flux_tol=flux_tol_bm,
                    max_iter=max_iter_bm
                )
            st.session_state["bm_result"] = bm_result

    if "bm_result" in st.session_state:
        bm_result: BenchmarkResult = st.session_state["bm_result"]
        st.divider()

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("计算 keff", f"{bm_result.keff_calc:.6f}")
        col_b.metric("参考 keff", f"{bm_result.keff_ref:.6f}")
        col_c.metric("keff误差", f"{bm_result.keff_error_pcm:.1f} pcm")
        col_d.metric("是否收敛", "✅ 是" if bm_result.converged else "❌ 否")

        if bm_result.keff_error_pcm < 100:
            st.success("🎉 keff误差 < 100 pcm，结果良好！")
        elif bm_result.keff_error_pcm < 500:
            st.warning("⚠️ keff误差在 100~500 pcm 之间，可接受范围")
        else:
            st.error("❌ keff误差 > 500 pcm，建议细化网格或检查参数")

        st.caption(f"外迭代次数: {bm_result.n_iterations}")

        if bm_result.x_centers.size > 0 and bm_result.phi1.size > 0:
            tab1, tab2 = st.tabs(["快群通量分布", "热群通量分布"])
            with tab1:
                fig1 = plot_2d_flux(bm_result.x_centers, bm_result.y_centers,
                                   bm_result.phi1, title="IAEA基准 - 快群通量 φ₁",
                                   cmap="plasma")
                st.pyplot(fig1)
            with tab2:
                fig2 = plot_2d_flux(bm_result.x_centers, bm_result.y_centers,
                                   bm_result.phi2, title="IAEA基准 - 热群通量 φ₂",
                                   cmap="viridis")
                st.pyplot(fig2)

elif page == "瞬态动力学模拟":
    st.header("⚡ 瞬态动力学模拟 (点堆动力学方程)")

    st.markdown("""
    点堆动力学方程组:
    - 功率方程: dP/dt = [(ρ(t)-β)/Λ]·P(t) + Σλᵢ·Cᵢ(t)
    - 先驱核方程: dCᵢ/dt = (βᵢ/Λ)·P(t) - λᵢ·Cᵢ(t)  (i=1~6)
    """)

    with st.sidebar:
        st.subheader("⏱️ 时间设置")
        t_end = st.number_input("总模拟时间 (s)", 0.1, 1000.0, 10.0, 0.1, key="tk_tend")
        dt_input = st.number_input("时间步长 (ms)", 0.01, 1000.0, 1.0, 0.01, key="tk_dt")
        dt = dt_input / 1000.0

        st.subheader("🧮 求解方法")
        solver_method = st.radio("数值方法", ["RK4 (四阶龙格-库塔)", "隐式欧拉"], key="tk_solver")

        st.subheader("⚛️ 核素选择")
        nuc_preset = st.selectbox(
            "缓发中子参数预设",
            ["U-235 (默认)", "U-238", "Pu-239", "自定义"],
            key="tk_nuc"
        )

        st.subheader("📐 中子代时间")
        Lambda = st.number_input("Λ (中子代时间, s)", 1e-6, 1e-2, 1e-4, format="%.2e", key="tk_Lambda")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎛️ 反应性扰动输入")
        reactivity_mode = st.radio(
            "反应性输入方式",
            ["阶跃插入", "线性爬升", "正弦振荡"],
            horizontal=True,
            key="tk_rmode"
        )

        if reactivity_mode == "阶跃插入":
            rc1, rc2 = st.columns(2)
            with rc1:
                rho_step = st.number_input(
                    "阶跃反应性幅度 (Δk/k)",
                    -0.01, 0.01, 0.001, 1e-5, format="%.5f", key="tk_rhostep"
                )
            with rc2:
                t_insert = st.number_input(
                    "插入时刻 (s)", 0.0, t_end, 1.0, 0.1, key="tk_tinsert"
                )
            rho_func = lambda t: reactivity_step(t, rho_step, t_insert)
            st.info(f"t < {t_insert:.1f}s: ρ = 0;  t ≥ {t_insert:.1f}s: ρ = {rho_step:.5f}")

        elif reactivity_mode == "线性爬升":
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1:
                t_start = st.number_input("起始时刻 (s)", 0.0, t_end, 1.0, 0.1, key="tk_tstart")
            with rc2:
                t_end_ramp = st.number_input("结束时刻 (s)", 0.0, t_end, 5.0, 0.1, key="tk_tendramp")
            with rc3:
                rho_start = st.number_input(
                    "起始反应性", -0.01, 0.01, 0.0, 1e-5, format="%.5f", key="tk_rhostart"
                )
            with rc4:
                rho_end = st.number_input(
                    "结束反应性", -0.01, 0.01, 0.002, 1e-5, format="%.5f", key="tk_rhoend"
                )
            if t_end_ramp <= t_start:
                st.error("结束时刻必须大于起始时刻!")
                rho_func = lambda t: 0.0
            else:
                rho_func = lambda t: reactivity_linear(t, t_start, t_end_ramp, rho_start, rho_end)

        elif reactivity_mode == "正弦振荡":
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1:
                amp = st.number_input(
                    "振荡幅度", 0.0, 0.01, 0.001, 1e-5, format="%.5f", key="tk_amp"
                )
            with rc2:
                freq = st.number_input(
                    "频率 (Hz)", 0.001, 10.0, 0.1, 0.01, key="tk_freq"
                )
            with rc3:
                phase = st.number_input(
                    "相位 (rad)", 0.0, 2 * np.pi, 0.0, 0.1, key="tk_phase"
                )
            with rc4:
                offset = st.number_input(
                    "直流偏置", -0.01, 0.01, 0.0, 1e-5, format="%.5f", key="tk_offset"
                )
            rho_func = lambda t: reactivity_sinusoidal(t, amp, freq, phase, offset)

    with col2:
        st.subheader("🔬 缓发中子参数 (可修改)")
        if nuc_preset == "U-235 (默认)":
            default_params = get_u235_params()
        elif nuc_preset == "U-238":
            default_params = get_u238_params()
        elif nuc_preset == "Pu-239":
            default_params = get_pu239_params()
        else:
            default_params = get_u235_params()

        beta_i_edit = np.zeros(6)
        lambda_i_edit = np.zeros(6)

        dn_table_data = []
        for i in range(6):
            cc1, cc2, cc3 = st.columns([1, 2, 2])
            with cc1:
                st.markdown(f"**组{i+1}**")
            with cc2:
                beta_i_edit[i] = st.number_input(
                    f"β{i+1}", 0.0, 0.01, float(default_params.beta_i[i]),
                    1e-6, format="%.6f", key=f"tk_b{i}"
                )
            with cc3:
                lambda_i_edit[i] = st.number_input(
                    f"λ{i+1} (s⁻¹)", 0.0, 10.0, float(default_params.lambda_i[i]),
                    1e-4, format="%.4f", key=f"tk_l{i}"
                )

        dn_params = DelayedNeutronParams(beta_i=beta_i_edit, lambda_i=lambda_i_edit)
        st.info(f"**总缓发中子份额 β = Σβᵢ = {dn_params.beta_total:.6f}**")

    st.divider()

    if st.button("🚀 开始瞬态动力学模拟", type="primary", use_container_width=True, key="tk_run"):
        with st.spinner("正在求解点堆动力学方程..."):
            if solver_method.startswith("RK4"):
                result = solve_point_kinetics_rk4(
                    rho_func, t_end=t_end, dt=dt,
                    Lambda=Lambda, dn_params=dn_params, P0=1.0
                )
            else:
                result = solve_point_kinetics_implicit_euler(
                    rho_func, t_end=t_end, dt=dt,
                    Lambda=Lambda, dn_params=dn_params, P0=1.0
                )
        st.session_state["kinetics_result"] = result

    if "kinetics_result" in st.session_state:
        result: KineticsResult = st.session_state["kinetics_result"]

        st.subheader("📊 关键参数汇总")
        kcol1, kcol2, kcol3, kcol4 = st.columns(4)
        kcol1.metric("最大功率倍数", f"{result.max_power_multiplier:.4f}")
        if result.steady_state_period is not None:
            ss_period = result.steady_state_period
            if abs(ss_period) > 1e4:
                kcol2.metric("稳态周期", f"≈ ∞ (稳定)")
            else:
                kcol2.metric("稳态周期 (s)", f"{ss_period:.4f}")
        else:
            kcol2.metric("稳态周期", "N/A")
        if result.asymptotic_period is not None:
            asymp = result.asymptotic_period
            if abs(asymp) > 1e4:
                kcol3.metric("渐近周期", f"≈ ∞ (稳定)")
            else:
                kcol3.metric("渐近周期 (s)", f"{asymp:.4f}")
        else:
            kcol3.metric("渐近周期", "N/A")
        kcol4.metric("时间步数", f"{result.n_steps}")

        kcol5, kcol6, kcol7, kcol8 = st.columns(4)
        kcol5.metric("中子代时间 Λ (s)", f"{result.Lambda:.2e}")
        kcol6.metric("总缓发份额 β", f"{result.beta_total:.6f}")
        kcol7.metric("β/Λ 比 (s⁻¹)", f"{result.beta_total / result.Lambda:.2e}")
        max_rho_idx = int(np.argmax(np.abs(result.reactivity)))
        kcol8.metric("最大|ρ(t)|", f"{result.reactivity[max_rho_idx]:.6f}")

        st.divider()

        show_log = st.checkbox("功率曲线使用对数坐标", value=False, key="tk_log")

        tab_p, tab_r, tab_c, tab_all = st.tabs([
            "📈 归一化功率 P(t)/P₀",
            "📉 反应性 ρ(t)",
            "⚛️ 先驱核浓度 Cᵢ(t)",
            "📋 综合视图"
        ])

        with tab_p:
            fig_p = plot_kinetics_power(
                result.time, result.power_normalized,
                log_scale=show_log,
                title=f"归一化功率曲线 (方法: {solver_method})"
            )
            st.pyplot(fig_p)

        with tab_r:
            fig_r = plot_kinetics_reactivity(result.time, result.reactivity)
            st.pyplot(fig_r)

        with tab_c:
            fig_c = plot_kinetics_precursors(
                result.time, result.precursors, result.lambda_i,
                normalize=True
            )
            st.pyplot(fig_c)

        with tab_all:
            fig_all = plot_kinetics_all(
                result.time, result.power_normalized,
                result.reactivity, result.precursors,
                result.lambda_i
            )
            st.pyplot(fig_all)

        st.subheader("📋 缓发中子参数表")
        table_data = {
            "组号": [f"组{i+1}" for i in range(6)],
            "βᵢ (份额)": [f"{result.beta_i[i]:.6f}" for i in range(6)],
            "λᵢ (s⁻¹)": [f"{result.lambda_i[i]:.4f}" for i in range(6)],
            "半衰期 (s)": [f"{np.log(2)/result.lambda_i[i]:.4f}" if result.lambda_i[i] > 0 else "∞" for i in range(6)]
        }
        st.dataframe(table_data, use_container_width=True, hide_index=True)

elif page == "蒙特卡洛模拟":
    st.header("🎲 一维蒙特卡洛中子输运模拟")

    st.markdown("""
    **蒙特卡洛输运原理:**
    - 模拟单个中子从出生到死亡的完整生命周期
    - 自由程长度按指数分布随机抽样: λ = -ln(ξ)/Σ_t
    - 反应类型按截面比例随机判定: 吸收、散射、裂变
    - 采用代际蒙特卡洛方法估计 keff
    """)

    with st.sidebar:
        st.subheader("🎲 蒙特卡洛参数")
        n_neutrons = st.slider("每代中子数", 100, 100000, 10000, 100, key="mc_n")
        n_generations = st.slider("模拟代数", 10, 500, 100, 5, key="mc_g")
        n_discard = st.slider("丢弃代数 (收敛前)", 0, 100, 20, 1, key="mc_d")
        n_flux_bins = st.slider("通量统计网格数", 10, 200, 50, 5, key="mc_bins")
        seed = st.number_input("随机数种子 (0=不设置)", 0, 999999, 42, 1, key="mc_seed")
        ma_window = st.slider("移动平均窗口", 1, 20, 5, 1, key="mc_ma")

        st.subheader("🏗️ 几何与材料 (单群)")
        dx_mc = st.slider("扩散方程网格间距 (cm)", 0.1, 5.0, 1.0, 0.1, key="mc_dx")

        bc_left_mc = st.selectbox("左边界", BOUNDARY_OPTIONS, index=0, key="mc_bc_l")
        bc_right_mc = st.selectbox("右边界", BOUNDARY_OPTIONS, index=0, key="mc_bc_r")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🏗️ 几何与材料定义")
        n_regions_mc = st.number_input("材料区域数", 1, 10, 2, 1, key="n_regions_mc")
        regions_mc = []
        current_x_mc = 0.0

        for i in range(n_regions_mc):
            st.markdown(f"**区域 {i+1}**")
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                width = st.number_input(f"宽度 (cm) - 区域{i+1}", 1.0, 200.0, 20.0, 1.0, key=f"w_mc_{i}")
            with rcol2:
                mat_name = st.selectbox(f"材料 - 区域{i+1}", MATERIAL_OPTIONS,
                                       index=i % len(MATERIAL_OPTIONS), key=f"m_mc_{i}")
            x_start = current_x_mc
            x_end = current_x_mc + width
            regions_mc.append(Region1D(x_start, x_end, mat_name))
            current_x_mc = x_end
            st.info(f"x ∈ [{x_start:.1f}, {x_end:.1f}] cm: {mat_name}")

    with col2:
        st.subheader("🧮 确定性扩散方程参数")
        k_tol_mc = st.number_input("keff收敛阈值", 1e-7, 1e-3, 1e-5, format="%.1e", key="mc_ktol")
        flux_tol_mc = st.number_input("通量收敛阈值", 1e-6, 1e-2, 1e-4, format="%.1e", key="mc_ftol")
        max_iter_mc = st.number_input("最大外迭代次数", 50, 2000, 500, 50, key="mc_maxiter")

        run_mc = st.checkbox("同时运行确定性扩散方程对比", value=True, key="mc_compare")

        seed_val = seed if seed > 0 else None

        if st.button("🚀 开始蒙特卡洛模拟", type="primary", use_container_width=True, key="run_mc"):
            bc_mc = BoundaryCondition(left=bc_left_mc, right=bc_right_mc)
            geom_mc = Geometry1D(regions=regions_mc, dx=dx_mc, bc=bc_mc, material_mode="1g")

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            current_keff_text = st.empty()

            def progress_cb(current_gen: int, total_gen: int, keff_gen: float):
                progress = current_gen / total_gen
                progress_bar.progress(progress)
                status_text.text(f"正在模拟第 {current_gen}/{total_gen} 代...")
                current_keff_text.metric("当前代 keff", f"{keff_gen:.6f}")

            with st.spinner("蒙特卡洛模拟进行中，请稍候..."):
                mc_result = run_monte_carlo_1d(
                    geom_mc,
                    n_neutrons_per_gen=n_neutrons,
                    n_generations=n_generations,
                    n_discard=n_discard,
                    n_flux_bins=n_flux_bins,
                    seed=seed_val,
                    progress_callback=progress_cb,
                )

            progress_bar.progress(1.0)
            status_text.text("✅ 蒙特卡洛模拟完成！")

            st.session_state["mc_result"] = mc_result
            st.session_state["geom_mc"] = geom_mc

            if run_mc:
                with st.spinner("正在运行确定性扩散方程..."):
                    det_result = criticality_1d_1g(
                        geom_mc,
                        k_tol=k_tol_mc,
                        flux_tol=flux_tol_mc,
                        max_iter=max_iter_mc,
                    )
                st.session_state["det_result_mc"] = det_result

    if "mc_result" in st.session_state:
        mc_result: MonteCarloResult = st.session_state["mc_result"]
        geom_mc = st.session_state["geom_mc"]

        st.divider()

        st.subheader("📊 蒙特卡洛结果统计")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("🎯 平均 keff", f"{mc_result.keff_mean:.6f}")
        col_b.metric("📏 标准差 σ", f"{mc_result.keff_std:.6f}")
        col_c.metric("📐 95% 置信区间",
                     f"[{mc_result.keff_ci_95[0]:.6f},\n {mc_result.keff_ci_95[1]:.6f}]")
        col_d.metric("⚠️ 相对误差", f"{mc_result.keff_relative_error*100:.4f}%")

        col_e, col_f, col_g, col_h = st.columns(4)
        col_e.metric("🔢 有效代数", f"{mc_result.effective_generations}")
        col_f.metric("⚛️ 总模拟中子数", f"{mc_result.total_neutrons_tracked:,}")
        col_g.metric("💥 总碰撞次数", f"{mc_result.total_collisions:,}")
        col_h.metric("☢️ 裂变源点数", f"{len(mc_result.fission_sites):,}")

        if "det_result_mc" in st.session_state:
            det_result = st.session_state["det_result_mc"]
            st.divider()
            st.subheader("⚖️ 蒙特卡洛 vs 扩散方程对比")
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            comp_col1.metric("蒙特卡洛 keff", f"{mc_result.keff_mean:.6f}")
            comp_col2.metric("扩散方程 keff", f"{det_result.keff:.6f}")
            keff_diff_pcm = abs(mc_result.keff_mean - det_result.keff) / det_result.keff * 1e6
            comp_col3.metric("keff 差异", f"{keff_diff_pcm:.1f} pcm")

            if keff_diff_pcm < 100:
                st.success(f"✅ keff 差异 < 100 pcm，两种方法一致性良好！")
            elif keff_diff_pcm < 500:
                st.warning(f"⚠️ keff 差异在 100~500 pcm 之间，可接受范围")
            else:
                st.error(f"❌ keff 差异 > 500 pcm，建议增加中子数或代数")

        st.divider()

        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 keff 收敛曲线",
            "📊 通量对比",
            "⚠️ 相对误差",
            "💥 碰撞分布"
        ])

        with tab1:
            fig_keff = plot_keff_convergence_mc(
                mc_result.keff_history,
                n_discard=mc_result.n_discard,
                keff_mean=mc_result.keff_mean,
                keff_std=mc_result.keff_std,
                window=ma_window,
                title="蒙特卡洛 keff 收敛曲线"
            )
            st.pyplot(fig_keff)

            if "det_result_mc" in st.session_state:
                det_result = st.session_state["det_result_mc"]
                st.info(f"📌 扩散方程 keff = {det_result.keff:.6f} (参考值)")

        with tab2:
            if "det_result_mc" in st.session_state:
                det_result = st.session_state["det_result_mc"]
                x_nodes, x_centers, _ = geom_mc.build_mesh()

                fig_flux = plot_flux_comparison(
                    mc_result.flux_centers,
                    mc_result.flux_mc,
                    x_centers,
                    det_result.phi,
                    title="蒙特卡洛 vs 扩散方程 通量分布对比"
                )
                st.pyplot(fig_flux)
            else:
                fig_flux_mc = plot_1d_profile(
                    mc_result.flux_centers,
                    mc_result.flux_mc,
                    title="蒙特卡洛中子通量分布",
                    xlabel="x (cm)"
                )
                st.pyplot(fig_flux_mc)

        with tab3:
            if "det_result_mc" in st.session_state:
                det_result = st.session_state["det_result_mc"]
                x_nodes, x_centers, _ = geom_mc.build_mesh()

                from scipy.interpolate import interp1d

                det_interp = interp1d(x_centers, det_result.phi, kind='linear',
                                     bounds_error=False, fill_value=0.0)
                det_at_mc = det_interp(mc_result.flux_centers)

                rel_error = compute_flux_relative_error(mc_result.flux_mc, det_at_mc)

                fig_err = plot_relative_error(
                    mc_result.flux_centers,
                    rel_error,
                    title="通量相对误差分布 (蒙特卡洛 vs 扩散方程)"
                )
                st.pyplot(fig_err)

                err_col1, err_col2, err_col3 = st.columns(3)
                err_col1.metric("平均相对误差", f"{np.mean(rel_error):.2f}%")
                err_col2.metric("最大相对误差", f"{np.max(rel_error):.2f}%")
                err_col3.metric("中位相对误差", f"{np.median(rel_error):.2f}%")
            else:
                st.info("请勾选'同时运行确定性扩散方程对比'以查看相对误差")

        with tab4:
            fig_col = plot_collision_histogram(
                mc_result.flux_centers,
                mc_result.collision_count,
                title="中子碰撞次数空间分布"
            )
            st.pyplot(fig_col)

st.divider()
st.caption("核反应堆中子扩散方程数值求解工具 | 基于 Streamlit + NumPy + SciPy + Matplotlib")
