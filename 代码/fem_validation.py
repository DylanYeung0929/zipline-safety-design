#!/usr/bin/env python3
"""
FEM铰接杆单元交叉验证
=====================
将钢缆离散为仅受轴向拉力的铰接杆单元(truss element),
以系统总势能最小化求解弹性悬链线平衡位形,
与打靶法解析解进行独立对比验证。
"""
import numpy as np
from scipy.optimize import minimize
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problem1_catenary import ElasticCatenary, G


def fem_truss_validation(W=600.0, H=20.0, L=615.0, R=1.2, E=110e9, n_elem=80):
    """
    FEM铰接杆单元法验证弹性悬链线解。

    将钢缆离散为n个铰接杆单元，每个单元仅受轴向拉/压力。
    节点总数: n+1 (编号0..n)
    未知量: 各节点位移 {u_i} = {x_i, y_i}, 但x_0,y_0和x_n,y_n固定。
    自由度为2*(n-1)。
    """

    rho_steel = 7850.0
    A_cable = R / rho_steel  # 钢缆截面积
    l0 = L / n_elem           # 单元无应力原长
    mass_per_node = R * l0    # 每个节点分担的钢缆质量

    # 节点初始猜测: 抛物线近似, 独立于打靶法
    x_init = np.linspace(0, W, n_elem + 1)
    y_parabola = H * (x_init / W) + 4 * (H * 0.1) * (x_init / W) * (1 - x_init / W)
    y_init = y_parabola

    # 自由度向量: [y_1, y_2, ..., y_{n-1}, x_1, ..., x_{n-1}]
    # (x_0, y_0) = (0, 0) 固定
    # (x_n, y_n) = (W, H) 固定
    dof_y0 = y_init[1:-1].copy()
    dof_x0 = x_init[1:-1].copy()
    dof0 = np.concatenate([dof_y0, dof_x0])

    def unpack(dof):
        """返回完整节点数组: (x_nodes, y_nodes)"""
        n_free = n_elem - 1
        y_free = dof[:n_free]
        x_free = dof[n_free:]
        x_nodes = np.concatenate([[0.0], x_free, [W]])
        y_nodes = np.concatenate([[0.0], y_free, [H]])
        return x_nodes, y_nodes

    def total_potential(dof):
        x_nodes, y_nodes = unpack(dof)

        # 弹性应变能: U = Σ (EA/(2*l0)) * (Δl)^2
        strain_energy = 0.0
        for e in range(n_elem):
            dx = x_nodes[e+1] - x_nodes[e]
            dy = y_nodes[e+1] - y_nodes[e]
            l_current = np.sqrt(dx**2 + dy**2)
            dl = l_current - l0
            strain_energy += 0.5 * E * A_cable / l0 * dl**2

        # 重力势能: V_g = Σ m_i * g * y_i
        grav_potential = 0.0
        for i in range(1, n_elem):  # 内节点
            grav_potential += mass_per_node * G * y_nodes[i]
        # 端点各承担半个单元质量
        grav_potential += 0.5 * mass_per_node * G * y_nodes[0]
        grav_potential += 0.5 * mass_per_node * G * y_nodes[n_elem]

        return strain_energy + grav_potential

    # BFGS优化
    result = minimize(total_potential, dof0, method='BFGS',
                      options={'maxiter': 15000, 'gtol': 1e-8})

    x_fem, y_fem = unpack(result.x)

    # 从FEM结果提取张力分布: T_e = EA * Δl_e / l0
    T_fem = np.zeros(n_elem + 1)
    for e in range(n_elem):
        dx = x_fem[e+1] - x_fem[e]
        dy = y_fem[e+1] - y_fem[e]
        l_current = np.sqrt(dx**2 + dy**2)
        dl = l_current - l0
        T_e = E * A_cable * dl / l0  # 单元轴力
        T_fem[e] = T_e
    T_fem[n_elem] = T_fem[n_elem-1]  # 末端近似

    # 与打靶法对比
    ec = ElasticCatenary(W=W, H=H, L=L, R=R, E=E)
    ec.solve_unloaded()
    x_fine = np.linspace(0, W, 200)
    y_cat = ec.unloaded_curve(x_fine)
    T_cat = ec.unloaded_tension(x_fine)

    # 插值FEM结果到打靶法网格
    y_fem_interp = np.interp(x_fine, x_fem, y_fem)
    T_fem_interp = np.interp(x_fine, x_fem, T_fem)

    # 误差统计
    dy_err = np.max(np.abs(y_fem_interp - y_cat))
    dT_err = np.max(np.abs(T_fem_interp - T_cat))

    print(f"\n{'='*60}")
    print(f"FEM交叉验证结果 (n={n_elem}铰接杆单元)")
    print(f"{'='*60}")
    print(f"  垂度 y(x) 最大偏差: {dy_err:.2e} m  (相对 {dy_err/np.max(np.abs(y_cat))*100:.4f}%)")
    print(f"  张力 T(x) 最大偏差: {dT_err:.2e} N  (相对 {dT_err/np.max(T_cat)*100:.4f}%)")
    print(f"  BFGS收敛: {result.success}, 迭代: {result.nit}")
    print(f"{'='*60}")

    return {
        'dy_max_err': dy_err,
        'dT_max_err': dT_err,
        'converged': result.success,
        'x_fem': x_fem, 'y_fem': y_fem, 'T_fem': T_fem,
        'x_cat': x_fine, 'y_cat': y_cat, 'T_cat': T_cat,
    }


if __name__ == "__main__":
    # 验证空载悬链线
    print("FEM铰接杆单元法 × 打靶法 独立交叉校验")
    result = fem_truss_validation(W=600.0, H=20.0, L=615.0, R=1.2, E=110e9, n_elem=80)

    # 额外: 不同H下的验证
    for H_test in [5, 10, 15]:
        r = fem_truss_validation(W=600.0, H=H_test, L=615.0, R=1.2, E=110e9, n_elem=80)
