"""
问题一：弹性悬链线静力学求解
=============================
方法：打靶法(Shooting) + 解析悬链线 + 弹性变形修正
输出：空载/载人钢缆形态、张力分布、弹性伸长量
"""
import numpy as np
from scipy.optimize import brentq, fsolve
from scipy.integrate import quad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os, sys


# ---- 论文图表统一样式 ----
for fname in ["PingFang SC", "Heiti SC", "STHeiti", "Songti SC", "Arial Unicode MS"]:
    if any(fname in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = fname
        break
plt.rcParams.update({
    "axes.unicode_minus": False,
    "figure.dpi": 180,
    "savefig.dpi": 320,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "axes.titleweight": "bold",
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
    "grid.alpha": 0.22,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
})

# ============ 物理常数 ============
G       = 9.81       # 重力加速度 (m/s^2)
RHO_FE  = 7850.0     # 钢材密度 (kg/m^3)
E_STEEL = 210e9      # 钢弹性模量 (Pa), 参考: GB/T 20118-2017

# ============ 输出目录 ============
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "图表")
os.makedirs(FIG_DIR, exist_ok=True)


class ElasticCatenary:
    """弹性悬链线求解器 — 打靶法 + 胡克定律修正"""

    def __init__(self, W: float, H: float, L: float, R: float, E: float = E_STEEL):
        self.W, self.H, self.L, self.R, self.E = W, H, L, R, E
        self.A  = R / RHO_FE            # 截面积 (m²)
        self.EA = E * self.A            # 轴向刚度 (N)
        self._solved_unloaded = False
        self._solved_loaded   = False

    # ---- 辅助函数 ----
    @staticmethod
    def _a(H0, R, g=G):
        return H0 / (R * g)

    @staticmethod
    def _y(x, x0, a):
        return a * np.cosh((x - x0) / a)

    @staticmethod
    def _yp(x, x0, a):
        return np.sinh((x - x0) / a)

    @staticmethod
    def _T(x, x0, a, H0):
        return H0 * np.cosh((x - x0) / a)

    # ---- 空载求解 ----
    def solve_unloaded(self):
        """打靶法: 找 H0 使弹性伸长后总长 = L"""
        if self._solved_unloaded:
            return

        W, H, L, R, g, EA = self.W, self.H, self.L, self.R, G, self.EA

        def stretched_length(H0):
            a  = self._a(H0, R, g)

            # y(W)-y(0)=H → a[cosh((W-x0)/a)-cosh(x0/a)] = H → 解 x0
            def f_x0(x0):
                return a * (np.cosh((W - x0) / a) - np.cosh(x0 / a)) - H

            x0 = brentq(f_x0, -W * 2, W * 3, xtol=1e-14)
            # 弹性伸长后弧长积分
            def integrand(x):
                ch = np.cosh((x - x0) / a)
                return ch / (1.0 + H0 * ch / EA)

            L_stretched, _ = quad(integrand, 0, W, limit=300, epsabs=1e-12)
            return L_stretched - L, x0

        # 初值估计: 近似抛物线 sag ≈ R*g*W^2/(8*H0)
        sag_guess = 0.08 * W
        H0_low = R * g * W ** 2 / (8 * sag_guess) * 0.3
        H0_high = R * g * W ** 2 / (8 * sag_guess) * 5.0

        # 确保 f(H0_low) 和 f(H0_high) 异号
        for _ in range(30):
            f_low, _ = stretched_length(H0_low)
            f_high, _ = stretched_length(H0_high)
            if f_low * f_high < 0:
                break
            if f_low > 0:
                H0_low *= 0.5
            else:
                H0_high *= 2.0

        self.H0 = brentq(lambda h: stretched_length(h)[0], H0_low, H0_high, xtol=1e-12)
        _, self.x0 = stretched_length(self.H0)
        self.a  = self._a(self.H0, R, g)
        self._solved_unloaded = True
        return self.H0, self.x0

    def unloaded_curve(self, x):
        if not self._solved_unloaded:
            self.solve_unloaded()
        y0_fix = self._y(0, self.x0, self.a)
        return self._y(x, self.x0, self.a) - y0_fix

    def unloaded_tension(self, x):
        if not self._solved_unloaded:
            self.solve_unloaded()
        return self._T(x, self.x0, self.a, self.H0)

    def unloaded_slope(self, x):
        if not self._solved_unloaded:
            self.solve_unloaded()
        return self._yp(x, self.x0, self.a)

    # ---- 载人求解 ----
    def solve_loaded(self, m_person: float, x_person: float):
        """载人: 分段悬链线 + 集中力跳跃条件"""
        if hasattr(self, 'm_person') and self.m_person == m_person and getattr(self, 'x_p', None) == x_person:
            return
        if not self._solved_unloaded:
            self.solve_unloaded()

        self.m_person = m_person
        self.x_p      = x_person
        W, H, L, R, g, EA = self.W, self.H, self.L, self.R, G, self.EA

        def stretched_length_loaded(H0):
            a = self._a(H0, R, g)

            def seg_params(vars):
                x01, x02 = vars
                y01 = -a * np.cosh(x01 / a)
                y02 = H - a * np.cosh((W - x02) / a)
                # 连续性: y1(xp) = y2(xp)
                c1 = a * np.cosh((x_person - x01) / a) + y01
                c2 = a * np.cosh((x_person - x02) / a) + y02
                eq1 = c1 - c2
                # 跳跃条件: y2' - y1' = mg/H0
                eq2 = np.sinh((x_person - x02) / a) - np.sinh((x_person - x01) / a) - m_person * g / H0
                return [eq1, eq2]

            x01, x02 = fsolve(seg_params, [self.x0 - 10, self.x0 + 10], maxfev=2000, xtol=1e-10)
            y01 = -a * np.cosh(x01 / a)
            y02 = H - a * np.cosh((W - x02) / a)

            def int_seg(lo, hi, x0):
                f = lambda x: np.cosh((x - x0) / a) / (1 + H0 * np.cosh((x - x0) / a) / EA)
                v, _ = quad(f, lo, hi, limit=300, epsabs=1e-12)
                return v

            L_stretched = int_seg(0, x_person, x01) + int_seg(x_person, W, x02)
            return L_stretched - L, x01, x02, y01, y02

        # 用空载 H0 做初值
        H0_low, H0_high = self.H0 * 0.3, self.H0 * 3.0
        for _ in range(30):
            fl, *_ = stretched_length_loaded(H0_low)
            fh, *_ = stretched_length_loaded(H0_high)
            if fl * fh < 0:
                break
            if fl > 0: H0_low *= 0.5
            else:       H0_high *= 2.0

        self.H0_ld = brentq(lambda h: stretched_length_loaded(h)[0], H0_low, H0_high, xtol=1e-12)
        _, self.x01, self.x02, self.y01, self.y02 = stretched_length_loaded(self.H0_ld)
        self.a_ld = self._a(self.H0_ld, R, g)
        self._solved_loaded = True

    def loaded_curve(self, x):
        if not self._solved_loaded:
            raise RuntimeError("请先调用 solve_loaded()")
        x = np.asarray(x)
        y = np.empty_like(x, dtype=float)
        m1, m2 = x <= self.x_p, x > self.x_p
        if m1.any():
            y[m1] = self._y(x[m1], self.x01, self.a_ld) + self.y01
            y[m1] -= (self._y(0, self.x01, self.a_ld) + self.y01)  # y(0)=0
        if m2.any():
            y[m2] = self._y(x[m2], self.x02, self.a_ld) + self.y02
            y[m2] -= (self._y(0, self.x01, self.a_ld) + self.y01)  # 同偏移
        return y

    def loaded_tension(self, x):
        if not self._solved_loaded:
            raise RuntimeError("请先调用 solve_loaded()")
        x = np.asarray(x)
        T = np.empty_like(x, dtype=float)
        m1, m2 = x <= self.x_p, x > self.x_p
        if m1.any(): T[m1] = self._T(x[m1], self.x01, self.a_ld, self.H0_ld)
        if m2.any(): T[m2] = self._T(x[m2], self.x02, self.a_ld, self.H0_ld)
        return T

    def loaded_slope(self, x):
        if not self._solved_loaded:
            raise RuntimeError("请先调用 solve_loaded()")
        x = np.asarray(x)
        yp = np.empty_like(x, dtype=float)
        m1, m2 = x <= self.x_p, x > self.x_p
        if m1.any(): yp[m1] = self._yp(x[m1], self.x01, self.a_ld)
        if m2.any(): yp[m2] = self._yp(x[m2], self.x02, self.a_ld)
        return yp

    @property
    def elastic_elongation(self):
        """弹性伸长量 ΔL = L_stretched - L_unstretched"""
        if not self._solved_unloaded:
            self.solve_unloaded()
        a, x0, H0 = self.a, self.x0, self.H0
        def unstretched(x):
            ch = np.cosh((x - x0) / a)
            return ch / (1 + H0 * ch / self.EA)
        L_unst, _ = quad(unstretched, 0, self.W, limit=300, epsabs=1e-12)
        L_st, _ = quad(lambda x: np.cosh((x - x0) / a), 0, self.W, limit=300, epsabs=1e-12)
        return L_st - L_unst

    @property
    def safety_factor(self):
        """钢缆强度安全系数 K = T_break / T_max
        T_break 参考 GB/T 20118-2017 钢丝绳标准, 抗拉强度 σ_b≈1770 MPa
        """
        SIGMA_B = 1770e6  # 钢丝绳公称抗拉强度 (Pa), 参考 GB/T 20118-2017
        T_break = SIGMA_B * self.A
        if not self._solved_unloaded:
            self.solve_unloaded()
        x_fine = np.linspace(0, self.W, 500)
        T_max = np.max(self.unloaded_tension(x_fine))
        return T_break / T_max


# ============ 敏感性分析: 高差 H 的影响 ============
def sensitivity_analysis(W=600.0, L=615.0, R=1.2, H_range=(5, 20), n_H=16):
    """计算不同高差下的关键力学指标"""
    H_vals = np.linspace(H_range[0], H_range[1], n_H)
    results = {"H": H_vals, "T_max": [], "T_min": [], "sag_max": [], "dL": []}

    for H in H_vals:
        ec = ElasticCatenary(W, H, L, R)
        ec.solve_unloaded()
        x_fine = np.linspace(0, W, 500)
        T = ec.unloaded_tension(x_fine)
        y = ec.unloaded_curve(x_fine)
        y_chord = H * x_fine / W                 # 弦线 (端点连线)
        sag_from_chord = y_chord - y             # 正值为钢缆低于弦线
        results["T_max"].append(np.max(T))
        results["T_min"].append(np.min(T))
        results["sag_max"].append(np.max(sag_from_chord))
        results["dL"].append(ec.elastic_elongation)

    for k in results:
        if k != "H":
            results[k] = np.array(results[k])
    return results


# ============ 图1: 空载 vs 载人 3D对比 ============
def plot_fig1_cable_3d(ec: ElasticCatenary, m_person=75.0, x_person=300.0):
    """图1: 空载与载人钢缆形变3D对比 (颜色=张力)"""
    ec.solve_unloaded()
    ec.solve_loaded(m_person, x_person)
    W = ec.W
    x = np.linspace(0, W, 400)

    y_unloaded = ec.unloaded_curve(x)
    T_unloaded = ec.unloaded_tension(x)
    y_loaded   = ec.loaded_curve(x)
    T_loaded   = ec.loaded_tension(x)

    fig = plt.figure(figsize=(14.5, 8.2))
    ax = fig.add_subplot(111, projection="3d")

    # 空载
    sc1 = ax.scatter(x, y_unloaded, T_unloaded / 1e3,
                     c=T_unloaded / 1e3, cmap="viridis", s=5, alpha=0.70, label="空载")
    # 载人
    sc2 = ax.scatter(x, y_loaded, T_loaded / 1e3,
                     c=T_loaded / 1e3, cmap="magma", s=7, alpha=0.82, label=f"载人 ({m_person}kg @ x={x_person}m)")

    ax.set_xlabel("水平位置 x (m)", fontsize=12)
    ax.set_ylabel("垂度 y (m)", fontsize=12)
    ax.set_zlabel("张力 T (kN)", fontsize=12)
    ax.set_title("空载与载人状态钢缆形变及张力分布", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    fig.colorbar(sc2, ax=ax, shrink=0.5, aspect=20, label="张力 (kN)")

    path = os.path.join(FIG_DIR, "fig01_cable_3d.pdf")
    fig.savefig(path, dpi=320, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(path.replace(".pdf", ".png"), dpi=320, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[图1] 已保存: {path}")
    return path


# ============ 图2: 高差敏感性曲面 ============
def plot_fig2_sensitivity(sa: dict):
    """图2: 高差 H 对张力和垂度的敏感性"""
    H = sa["H"]
    W = 600.0  # 已知跨度
    x_positions = np.linspace(0, W, 21)
    H_grid, X_grid = np.meshgrid(H, x_positions)

    # 计算每个 (H, x) 的垂度和张力
    Z_sag  = np.zeros_like(H_grid)
    Z_ten  = np.zeros_like(H_grid)

    for i, Hi in enumerate(H):
        ec = ElasticCatenary(W, Hi, 615.0, 1.2)
        ec.solve_unloaded()
        for j, xi in enumerate(x_positions):
            y_abs = ec.unloaded_curve(xi)
            y_chord = Hi * xi / W
            Z_sag[j, i] = y_chord - y_abs       # 弦线以下为正
            Z_ten[j, i] = ec.unloaded_tension(xi)

    fig = plt.figure(figsize=(16.5, 6.6))

    # 子图1: 垂度
    ax1 = fig.add_subplot(121, projection="3d")
    surf1 = ax1.plot_surface(H_grid, X_grid, Z_sag, cmap="YlGnBu", edgecolor="none", alpha=0.9)
    ax1.set_xlabel("高差 H (m)")
    ax1.set_ylabel("水平位置 x (m)")
    ax1.set_zlabel("垂度 y (m)")
    ax1.set_title("高差对垂度分布的影响")
    ax1.view_init(elev=25, azim=-60)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=20, label="垂度 (m)")

    # 子图2: 张力
    ax2 = fig.add_subplot(122, projection="3d")
    surf2 = ax2.plot_surface(H_grid, X_grid, Z_ten / 1e3, cmap="magma", edgecolor="none", alpha=0.9)
    ax2.set_xlabel("高差 H (m)")
    ax2.set_ylabel("水平位置 x (m)")
    ax2.set_zlabel("张力 T (kN)")
    ax2.set_title("高差对张力分布的影响")
    ax2.view_init(elev=25, azim=-60)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=20, label="张力 (kN)")

    path = os.path.join(FIG_DIR, "fig02_sensitivity.pdf")
    fig.savefig(path, dpi=320, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(path.replace(".pdf", ".png"), dpi=320, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[图2] 已保存: {path}")
    return path


# ============ 表1 数据输出 ============
def table1_data(sa: dict):
    """输出表1: 不同高差下关键力学指标"""
    print("\n" + "=" * 90)
    print("表1: 不同高差下钢缆关键力学指标")
    print("=" * 90)
    print(f"{'H(m)':>6s}  {'T_max(kN)':>12s}  {'T_min(kN)':>12s}  {'sag_max(m)':>12s}  {'ΔL(m)':>10s}")
    print("-" * 60)
    for i, H in enumerate(sa["H"]):
        print(f"{H:6.1f}  {sa['T_max'][i]/1e3:12.2f}  {sa['T_min'][i]/1e3:12.2f}  "
              f"{sa['sag_max'][i]:12.3f}  {sa['dL'][i]:10.4f}")
    print("-" * 60)


# ============ 主程序 ============
if __name__ == "__main__":
    # 景区参数 (问题三的典型值)
    W, H0_val, L, R = 600.0, 12.5, 615.0, 1.2
    print(f"参数: W={W}m, H={H0_val}m, L={L}m, R={R}kg/m")
    print(f"钢缆截面积 A = {R/RHO_FE*1e6:.2f} mm²")
    print(f"轴向刚度 EA = {E_STEEL*R/RHO_FE/1e6:.2f} MN\n")

    # 求解
    ec = ElasticCatenary(W, H0_val, L, R)
    H0_sol, x0_sol = ec.solve_unloaded()
    print(f"空载解: H0 = {H0_sol/1e3:.2f} kN, x0 = {x0_sol:.2f} m")
    print(f"  a = {ec.a:.1f} m, ΔL_elastic = {ec.elastic_elongation*1e3:.1f} mm")

    ec.solve_loaded(m_person=75.0, x_person=300.0)
    print(f"载人解(m=75kg, xp=300m): H0 = {ec.H0_ld/1e3:.2f} kN")

    # 敏感性分析
    sa = sensitivity_analysis(W, L, R)
    table1_data(sa)

    # 生成图表
    plot_fig1_cable_3d(ec)
    plot_fig2_sensitivity(sa)
    print("\n问题一全部图表已生成。")
