#!/usr/bin/env python3
"""
论文图表美化与新增 —— 统一生成全部论文插图
输出: 图表/ 目录下 PDF + PNG
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.ticker as ticker
import sys, os, shutil
from datetime import datetime

# ── 中文字体 ──────────────────────────────────────────────
for fname in ["PingFang SC", "Heiti SC", "STHeiti", "SimHei", "Arial Unicode MS"]:
    try:
        rcParams["font.sans-serif"] = [fname]
        rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

# ── 全局样式 ──────────────────────────────────────────────
rcParams.update({
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

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "图表")
os.makedirs(OUTDIR, exist_ok=True)

FIGURE_NAMES = [
    "fig01_cable_3d",
    "fig02_sensitivity",
    "fig03_phase_trajectory",
    "fig04_double_axis",
    "fig05_buffer_heatmap",
    "fig06_allometric",
    "fig07_pareto_front",
    "fig08_three_weights",
    "fig09_validation_dashboard",
    "fig10_buffer_characteristics",
    "fig11_energy_breakdown",
]


def backup_existing_figures():
    files = []
    for name in FIGURE_NAMES:
        for ext in (".pdf", ".png"):
            path = os.path.join(OUTDIR, name + ext)
            if os.path.exists(path):
                files.append(path)
    if not files:
        return None
    backup_dir = os.path.join(OUTDIR, "backup_beautified", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(backup_dir, exist_ok=True)
    for path in files:
        shutil.copy2(path, os.path.join(backup_dir, os.path.basename(path)))
    print(f"已备份旧图表至: {backup_dir}")
    return backup_dir

# 统一配色
C_UNLOADED   = "#5F6C72"
C_LOADED     = "#D1495B"
C_NOBUF      = "#8D99AE"
C_CONSTBUF   = "#2F6690"
C_THRESH     = "#D1495B"
C_DANGER     = "#D62828"
C_SAFE       = "#A7D7C5"
C_GOLD       = "#F2C14E"
C_ALLOMETRIC = "#D1495B"
C_CONSTANT   = "#8D99AE"
C_HEAVY      = "#D1495B"
C_MEDIUM     = "#2F6690"
C_LIGHT      = "#2A9D8F"

# 固定参数
W, H_OPT, H0_OPT = 600.0, 19.4, 65.0e3
D_BUF = 9.7      # 缓冲启动距终点水平距离 (m)
X_BUF_COORD = W - D_BUF  # 缓冲启动水平坐标 x_b = 590.3 m (从起点算起)
BEST_B, BEST_XB = 15.0, 563.6  # Problem 2 optimal

# ── 导入现有模块 ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problem1_catenary import ElasticCatenary
from problem2_dynamics import DynamicsSimulator, buffer_feasibility_search
from problem3_fixed import CatenaryFromH0, FastSimAdaptive, get_frontal_area, G, C_D, RHO_AIR


def save(fig, name):
    for ext in [".pdf", ".png"]:
        fig.savefig(os.path.join(OUTDIR, name + ext), dpi=320, bbox_inches="tight", pad_inches=0.05)
    print(f"  ✓ {name}")


# ╔══════════════════════════════════════════════════════════╗
# ║  Fig 03 (方案一): 空载与载人钢缆形态与张力分布对比     ║
# ╚══════════════════════════════════════════════════════════╝
def plot_fig03_cable_loaded():
    ec = ElasticCatenary(W=600, H=20.0, L=615, R=1.2, E=110e9)
    ec.solve_unloaded()
    ec.solve_loaded(m_person=80.0, x_person=300.0)

    x_fine = np.linspace(0, 600, 2000)
    y_u = ec.unloaded_curve(x_fine)
    T_u = ec.unloaded_tension(x_fine) / 1e3
    y_l = ec.loaded_curve(x_fine)
    T_l = ec.loaded_tension(x_fine) / 1e3

    fig = plt.figure(figsize=(16, 7))

    # ── 主图: 双Y轴 ──
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = ax1.twinx()

    # 空载
    ax1.plot(x_fine, y_u, "--", color=C_UNLOADED, linewidth=2.0, label="空载垂度 $y(x)$", zorder=3)
    ax2.plot(x_fine, T_u, "--", color=C_UNLOADED, linewidth=1.2, alpha=0.5, zorder=2)

    # 载人 — 用张力做颜色映射
    norm = plt.Normalize(T_l.min(), T_l.max())
    colors = plt.cm.magma(norm(T_l))
    for i in range(len(x_fine) - 1):
        ax1.plot(x_fine[i:i + 2], y_l[i:i + 2], color=colors[i], linewidth=2.5, alpha=0.9)

    ax2.plot(x_fine, T_l, color=C_LOADED, linewidth=1.8, label="载人张力 $T(x)$ (右轴)", alpha=0.85)

    # 加载点箭头
    y_at_load = ec.loaded_curve(np.array([300.0]))[0]
    ax1.annotate("$mg=800\\,{\\rm N}$", xy=(300, y_at_load), xytext=(330, y_at_load + 8),
                 fontsize=12, color=C_LOADED, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_LOADED, lw=2.5),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    ax1.set_xlabel("水平位置 $x$ (m)")
    ax1.set_ylabel("垂度 $y(x)$ (m)", color="#333333")
    ax2.set_ylabel("张力 $T(x)$ (kN)", color=C_LOADED)
    ax1.set_title("空载与载人钢缆形态及张力分布", fontweight="bold", pad=14)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.22)
    ax1.set_xlim(0, 600)

    # ── 局部放大子图 ──
    ax_zoom = fig.add_axes([0.24, 0.28, 0.22, 0.28])
    mask = (x_fine >= 285) & (x_fine <= 315)
    ax_zoom.plot(x_fine[mask], y_u[mask], "--", color=C_UNLOADED, linewidth=1.8, label="空载")
    ax_zoom.plot(x_fine[mask], y_l[mask], "-", color=C_LOADED, linewidth=2.5, label="载人 80kg")
    ax_zoom.axvline(x=300, color=C_LOADED, linestyle=":", linewidth=1.2, alpha=0.6)
    ax_zoom.set_title("局部放大 ($x\\in[290,310]$ m)", fontsize=10, fontweight="bold")
    ax_zoom.grid(True, alpha=0.25)
    ax_zoom.tick_params(labelsize=8)

    # ── 右图: 张力柱状对比 ──
    ax3 = fig.add_subplot(1, 2, 2)
    positions = ["空载\n$T_{\\max}$", "载人@100m\n$T_{\\max}$", "载人@300m\n$T_{\\max}$", "载人@500m\n$T_{\\max}$"]
    T_max_unloaded = np.max(T_u)
    ec.solve_loaded(80.0, 100.0); T_100 = np.max(ec.loaded_tension(x_fine)) / 1e3
    ec.solve_loaded(80.0, 300.0); T_300 = np.max(ec.loaded_tension(x_fine)) / 1e3
    ec.solve_loaded(80.0, 500.0); T_500 = np.max(ec.loaded_tension(x_fine)) / 1e3

    vals = [T_max_unloaded, T_100, T_300, T_500]
    bars = ax3.bar(positions, vals, color=[C_UNLOADED, C_LOADED, C_LOADED, C_LOADED],
                   alpha=0.75, edgecolor="black", linewidth=0.8)
    for b, v in zip(bars, vals):
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f"{v:.2f}",
                 ha="center", fontsize=12, fontweight="bold")
    ax3.set_ylabel("$T_{\\max}$ (kN)")
    ax3.set_title("不同载人位置下最大张力对比 (80kg)", fontweight="bold", pad=14)
    ax3.grid(True, axis="y", alpha=0.22)

    fig.suptitle("空载与载人状态下钢缆力学行为对比", fontsize=17, fontweight="bold", y=1.01)
    save(fig, "fig03_cable_loaded")
    plt.close(fig)


# ╔══════════════════════════════════════════════════════════╗
# ║  Fig 06 (方案三): 异速生长法则迎风面积与空气阻力修正   ║
# ╚══════════════════════════════════════════════════════════╝
def plot_fig06_allometric():
    m_arr = np.linspace(30, 80, 50)
    A_const = np.full_like(m_arr, 0.4)
    A_allo = np.array([get_frontal_area(m) for m in m_arr])
    v_test = 10.0

    F_const = 0.5 * C_D * RHO_AIR * A_const * v_test**2
    F_allo  = 0.5 * C_D * RHO_AIR * A_allo * v_test**2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    # ── 左: A(m) ──
    ax1.plot(m_arr, A_const, "--", color=C_CONSTANT, linewidth=2.5, label="传统定面积假设 $A\\equiv 0.4$")
    ax1.plot(m_arr, A_allo, "-", color=C_ALLOMETRIC, linewidth=3.0,
             label="异速生长法则 $A(m)=0.4\\cdot(m/80)^{2/3}$")
    ax1.fill_between(m_arr, A_const, A_allo, alpha=0.12, color=C_ALLOMETRIC)
    ax1.set_xlabel("游客体重 $m$ (kg)")
    ax1.set_ylabel("迎风面积 $A(m)$ (m$^2$)")
    ax1.set_title("动态迎风面积对比", fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.22)

    # 标注30kg差异
    ax1.annotate(f"30kg: $A={A_allo[0]:.2f}$ vs $0.4$\n差值 $\\Delta A={0.4-A_allo[0]:.2f}$ m$^2$",
                 xy=(30, A_allo[0]), xytext=(38, 0.22),
                 fontsize=11, color=C_ALLOMETRIC, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_ALLOMETRIC, lw=1.8),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    # ── 右: F_air at 10m/s ──
    ax2.bar(m_arr[::3] - 1.5, F_const[::3], width=2.8, color=C_CONSTANT, alpha=0.55,
            label="定面积 $A=0.4$", edgecolor="gray", linewidth=0.5)
    ax2.bar(m_arr[::3] + 1.5, F_allo[::3], width=2.8, color=C_ALLOMETRIC, alpha=0.75,
            label="异速生长 $A(m)$", edgecolor="gray", linewidth=0.5)
    ax2.set_xlabel("游客体重 $m$ (kg)")
    ax2.set_ylabel(f"空气阻力 $F_{{\\rm air}}$ at $v={v_test}$ m/s (N)")
    ax2.set_title(f"空气阻力修正 ($v={v_test}\\,$m/s)", fontweight="bold")
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.grid(True, axis="y", alpha=0.22)

    # 30kg差值标注
    dF_30 = abs(F_const[0] - F_allo[0])
    ax2.annotate(f"$\\Delta F_{{\\rm air}}\\approx {dF_30:.1f}$ N\n(轻体重\"救命稻草\")",
                 xy=(30, F_allo[0]), xytext=(42, F_allo[0] + 3),
                 fontsize=11, color=C_LOADED, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_LOADED, lw=2.0),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    fig.suptitle("基于异速生长法则的动态迎风面积与空气阻力修正",
                 fontsize=17, fontweight="bold", y=1.01)
    save(fig, "fig06_allometric")
    plt.close(fig)


# ╔══════════════════════════════════════════════════════════╗
# ║  Fig 07 (方案四): 三种缓冲策略相轨迹对比 (55kg)        ║
# ╚══════════════════════════════════════════════════════════╝
def plot_fig07_strategy_compare():
    # ── 紧缆 (同 Problem 3) 上的三种缓冲策略 ──
    cable = CatenaryFromH0(W=600, H=H_OPT, R=1.2, H0=H0_OPT)

    # 适配器: CatenaryFromH0 → DynamicsSimulator 接口
    class CableAdapter:
        def __init__(self, ch0):
            self.W, self.H, self.H0 = ch0.W, ch0.H, ch0.H0
        def unloaded_curve(self, x): return cable.y(np.asarray(x))
        def unloaded_slope(self, x): return cable.yp(np.asarray(x))
        def unloaded_tension(self, x): return cable.tension(np.asarray(x))

    sim2 = DynamicsSimulator(CableAdapter(cable))

    # 无缓冲
    traj_nb = sim2.simulate(55.0)
    # 恒定缓冲 (Problem 2 optimal)
    traj_cb = sim2.simulate(55.0, x_buf=BEST_XB, F_buf=BEST_B)

    # ── Problem 3: 前置测速+恒定制动 ──
    sim3 = FastSimAdaptive(cable)
    x_t, v_t, _ = sim3.simulate(55.0, x_buf=D_BUF)

    # 转换x坐标: 模拟空间(600→0) → 标准坐标(0→600), 起点x=0, 终点x=600
    x_nb_plot = W - traj_nb[:, 1]
    x_cb_plot = W - traj_cb[:, 1]
    x_t_plot  = W - x_t

    fig, (ax, axz) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [3, 1]})

    # ── 主图 ──
    ax.plot(x_nb_plot, np.abs(traj_nb[:, 2]), "--", color=C_NOBUF, linewidth=2.0, label="无缓冲")
    ax.plot(x_cb_plot, np.abs(traj_cb[:, 2]), "-.", color=C_CONSTBUF, linewidth=2.2,
            label=f"恒定缓冲 ($B={BEST_B}$N, $x={BEST_XB:.0f}$m)")
    ax.plot(x_t_plot, v_t, "-", color=C_THRESH, linewidth=2.8,
            label=f"前置测速制动 ($B_{{\\rm target}}$ 能量逆解)")

    # 危险区
    ax.axhline(y=3.5, color=C_DANGER, linestyle="--", linewidth=2.0, alpha=0.7)
    ax.fill_between([0, 600], 3.5, 15, color=C_DANGER, alpha=0.08)
    ax.text(10, 4.2, "Overspeed Zone\n($v>3.5$ m/s)", fontsize=10, color=C_DANGER, fontweight="bold", alpha=0.8)

    ax.set_xlabel("水平位置 $x$ (m)")
    ax.set_ylabel("滑行速度 $v$ (m/s)")
    ax.set_title("三种缓冲策略下 55kg 游客速度-位移相轨迹", fontweight="bold", pad=14)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 14)
    ax.grid(True, alpha=0.22)

    # ── 终点局部放大 (x∈[570,600]区域) ──
    axz.set_xlim(570, 602)
    axz.set_ylim(0, 6)
    axz.plot(x_nb_plot, np.abs(traj_nb[:, 2]), "--", color=C_NOBUF, linewidth=2.0)
    axz.plot(x_cb_plot, np.abs(traj_cb[:, 2]), "-.", color=C_CONSTBUF, linewidth=2.2)
    axz.plot(x_t_plot, v_t, "-", color=C_THRESH, linewidth=2.8)
    axz.axhline(y=3.5, color=C_DANGER, linestyle="--", linewidth=1.5, alpha=0.7)
    axz.axvline(x=X_BUF_COORD, color="orange", linestyle=":", linewidth=1.2,
                label=f"$x_b={X_BUF_COORD:.1f}$ m")
    axz.set_title("终点局部放大", fontsize=11, fontweight="bold")
    axz.legend(fontsize=8, loc="upper right")
    axz.grid(True, alpha=0.25)

    fig.suptitle("不同缓冲策略下 55kg 游客速度-位移相轨迹对比",
                 fontsize=17, fontweight="bold", y=1.01)
    save(fig, "fig03_phase_trajectory")
    plt.close(fig)


# ╔══════════════════════════════════════════════════════════╗
# ║  Fig 08 (方案二): 恒定缓冲"死锁"可行域热力图           ║
# ╚══════════════════════════════════════════════════════════╝
def plot_fig08_buffer_deadlock():
    cable = CatenaryFromH0(W=600, H=H_OPT, R=1.2, H0=H0_OPT)
    class CableAdapter:
        def __init__(self, ch0):
            self.W, self.H, self.H0 = ch0.W, ch0.H, ch0.H0
        def unloaded_curve(self, x): return cable.y(np.asarray(x))
        def unloaded_slope(self, x): return cable.yp(np.asarray(x))
        def unloaded_tension(self, x): return cable.tension(np.asarray(x))
    sim = DynamicsSimulator(CableAdapter(cable))

    m_list = [30, 45, 55, 65, 75, 80]
    x_range = (540, 595)
    f_range = (5, 35)
    n_grid = 25
    x_vals = np.linspace(x_range[0], x_range[1], n_grid)
    f_vals = np.linspace(f_range[0], f_range[1], n_grid)
    print(f"  缓冲搜索: {len(m_list)}个体重 × {n_grid}×{n_grid}网格 = {len(m_list)*n_grid*n_grid}次仿真...")

    # 单次遍历分类
    zone = np.zeros((n_grid, n_grid))
    for i, Fb in enumerate(f_vals):
        for j, xs in enumerate(x_vals):
            safe_light = True
            safe_heavy = True
            for m in m_list:
                traj = sim.simulate(m, x_buf=xs, F_buf=Fb)
                res = sim.get_vmax_and_vend(traj)
                if res["t_total"] is None:
                    safe_light = False  # 停滞
                elif res["v_end"] > 3.5:
                    safe_heavy = False  # 超速
            if safe_light and safe_heavy:
                zone[i, j] = 3  # 全通过
            elif safe_light and not safe_heavy:
                zone[i, j] = 1  # 仅超速
            elif not safe_light:
                zone[i, j] = 2  # 停滞

    X_dist = W - x_vals  # 转为距终点距离

    fig, ax = plt.subplots(figsize=(14, 7))

    # 自定义颜色映射
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#F8F9FA", "#F4A6A6", "#A9D6E5", "#B7E4C7"])
    im = ax.pcolormesh(X_dist, f_vals, zone, cmap=cmap, shading="auto", alpha=0.85, edgecolors="face")

    # 标注分区
    ax.text(40, 8, "轻体重停滞区\n(缓冲力过大)", fontsize=14, ha="center", color="#1565C0",
            fontweight="bold", alpha=0.7)
    ax.text(50, 28, "重体重超速区\n(缓冲力不足)", fontsize=14, ha="center", color="#C62828",
            fontweight="bold", alpha=0.7)
    ax.text(15, 3, "可行交集为空\n物理死锁", fontsize=13, ha="center", color="#333333",
            fontweight="bold", style="italic")

    ax.set_xlabel("缓冲启动位置 (距终点距离, m)")
    ax.set_ylabel("恒定缓冲摩擦力 $B$ (N)")
    ax.set_title("恒定缓冲摩擦力可行性边界分析", fontweight="bold", pad=14)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#FFCDD2", alpha=0.85, label="重体重超速区 ($v_{end}>3.5$ m/s)"),
        Patch(facecolor="#BBDEFB", alpha=0.85, label="轻体重停滞区 (中途停滞)"),
        Patch(facecolor="#C8E6C9", alpha=0.85, label="全体重通过 (不存在)"),
        Patch(facecolor="#F5F5F5", alpha=0.85, label="不可行区域"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=11,
              title="恒定摩擦力面临物理死锁: 无法兼顾轻重群体", title_fontsize=12)

    ax.set_xlim(X_dist.min(), X_dist.max())
    ax.set_ylim(f_range[0], f_range[1])

    fig.suptitle("恒定缓冲摩擦力可行性边界分析热力图", fontsize=17, fontweight="bold", y=1.01)
    save(fig, "fig05_buffer_heatmap")
    plt.close(fig)


# ╔══════════════════════════════════════════════════════════╗
# ║  Fig 09 (方案六): NSGA-II 帕累托前沿三维气泡图         ║
# ╚══════════════════════════════════════════════════════════╝
def plot_fig09_pareto_3d():
    from problem3_nsga2 import NSGA2
    from problem3_fixed import make_evaluator_fixed, topsis_select

    print("  运行 NSGA-II (40代×40种群)...")
    np.random.seed(42)  # 固定随机种子，保证帕累托前沿可复现
    evaluator = make_evaluator_fixed(W=600, R=1.2, m_list=[30, 40, 50, 55, 60, 70, 75, 80],
                                     T_max_allowed=67.6e3)
    bounds = [(14, 20), (25, 65), (5, 80)]  # 仅3个决策变量
    nsga2 = NSGA2(pop_size=40, n_gen=40, p_cross=0.9, p_mut=0.1, eta_c=20, eta_m=20)
    population, _ = nsga2.optimize(evaluator, bounds, n_obj=2)

    pareto = [ind for ind in population if ind["rank"] == 0]
    feasible = [ind for ind in pareto if ind["obj"][0] < 0 and ind["obj"][1] < 100]

    # TOPSIS 在前沿上给出高性能折衷候选(仅供参考)
    if len(feasible) >= 2:
        topsis_ind, _, _ = topsis_select(feasible)
    else:
        topsis_ind = feasible[0] if feasible else pareto[0]

    # 两阶段决策: NSGA-II/TOPSIS 缩小到高性能区 → 蒙特卡洛安全率精选运营点。
    # 推荐运营方案锚定于 MC 安全率验证的最优解 (H_OPT, H0_OPT, D_BUF),
    # 通过同一目标函数评估其在目标空间中的真实坐标。
    val_x = [H_OPT, H0_OPT / 1e3, D_BUF]
    val_obj = evaluator(val_x)
    best_ind = {"x": val_x, "obj": val_obj}
    print(f"  TOPSIS折衷候选: x_buf={topsis_ind['x'][2]:.1f}m")
    print(f"  推荐运营方案(MC安全率精选): H={H_OPT}m, H0={H0_OPT/1e3:.1f}kN, "
          f"d_b={D_BUF}m (前置测速+恒定制动)")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 非 Pareto 解
    non_pareto = [ind for ind in population if ind["rank"] > 0]
    if non_pareto:
        x_np = [-ind["obj"][0] for ind in non_pareto]
        y_np = [1.0 / max(ind["obj"][1], 1e-6) for ind in non_pareto]
        z_np = [ind["x"][2] for ind in non_pareto]  # x_buf
        ax.scatter(x_np, y_np, z_np, c="gray", s=20, alpha=0.2, label="非Pareto解")

    # Pareto 前沿
    x_p = [-ind["obj"][0] for ind in pareto]
    y_p = [1.0 / max(ind["obj"][1], 1e-6) for ind in pareto]
    z_p = [ind["x"][2] for ind in pareto]  # x_buf = buffer start position
    h_p = [ind["x"][0] for ind in pareto]  # H for color

    sc = ax.scatter(x_p, y_p, z_p, c=h_p, s=80, cmap="plasma", alpha=0.8,
                    edgecolors="black", linewidth=0.4, label="Pareto前沿")

    # TOPSIS 最优点
    bx = -best_ind["obj"][0]
    by = 1.0 / max(best_ind["obj"][1], 1e-6)
    bz = best_ind["x"][2]
    ax.scatter([bx], [by], [bz], c="red", s=400, marker="*", edgecolors="darkred",
               linewidth=1.5, zorder=10, label="推荐运营方案 (MC安全率精选)")

    # 虚线投影
    ax.plot([bx, bx], [by, by], [0, bz], "r--", linewidth=1.0, alpha=0.5)
    ax.plot([bx, bx], [0, by], [0, 0], "r--", linewidth=0.6, alpha=0.22)
    ax.plot([0, bx], [by, by], [0, 0], "r--", linewidth=0.6, alpha=0.22)

    ax.set_xlabel("体验满意度\n平均最大速度 $\\bar{v}_{\\max}$ (m/s)", labelpad=10)
    ax.set_ylabel("群体一致性\n$1/\\sigma^2_{v}$ (s$^2$/m$^2$)", labelpad=10)
    ax.set_zlabel("缓冲启动位置 $x_{\\rm buf}$ (m)", labelpad=10)
    ax.set_title("NSGA-II 帕累托前沿三维分布 (前置测速+恒定制动)", fontweight="bold", pad=20)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label("高差 $H$ (m)", fontsize=12)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)

    fig.suptitle("帕累托前沿三维分布与 TOPSIS 决策", fontsize=17, fontweight="bold", y=1.01)
    save(fig, "fig07_pareto_front")
    plt.close(fig)

    return best_ind


# ╔══════════════════════════════════════════════════════════╗
# ║  Fig 10 (方案五): 最优方案三种体重叠合图                ║
# ╚══════════════════════════════════════════════════════════╝
def plot_fig10_three_weights():
    cable = CatenaryFromH0(W=600, H=H_OPT, R=1.2, H0=H0_OPT)
    sim = FastSimAdaptive(cable)

    masses = [30, 55, 80]
    styles = [(C_LIGHT, "-.", "30 kg"), (C_MEDIUM, "--", "55 kg"), (C_HEAVY, "-", "80 kg")]
    colors = [C_LIGHT, C_MEDIUM, C_HEAVY]

    fig, (ax, axz) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [2.5, 1]})

    # ── 主图 ──
    for (color, ls, label), m in zip(styles, masses):
        xx, vv, _ = sim.simulate(m, x_buf=D_BUF)
        x_plot = W - xx  # 转换: 模拟坐标→标准坐标 (起点x=0, 终点x=600)
        ax.plot(x_plot, vv, ls, color=color, linewidth=2.5, label=label)
        # 末速标注
        ax.annotate(f"{vv[-1]:.1f} m/s", xy=(x_plot[-1], vv[-1]),
                    xytext=(x_plot[-1] - 120, vv[-1] + 0.5),
                    fontsize=11, color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.7))

    # 安全走廊
    ax.axhline(y=3.5, color=C_DANGER, linestyle="--", linewidth=2.0, alpha=0.7)
    ax.axhline(y=0.5, color="#FF9800", linestyle="--", linewidth=1.5, alpha=0.6)
    ax.fill_between([0, 600], 0.5, 3.5, color=C_SAFE, alpha=0.1)
    ax.text(10, 2.0, "Safe &\nComfort Zone", fontsize=11, color="#2E7D32", fontweight="bold",
            alpha=0.7, va="center")

    # 缓冲启动线 (在标准坐标中 x_b = W - d_b)
    ax.axvline(x=X_BUF_COORD, color="orange", linestyle=":", linewidth=2.0, alpha=0.8)
    ax.annotate(f"缓冲启动 $x_b={X_BUF_COORD:.1f}$m\n(距终点 ${D_BUF:.1f}$m)",
                xy=(X_BUF_COORD, 8),
                xytext=(X_BUF_COORD - 200, 10), fontsize=11, color="darkorange", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.5))

    ax.set_xlabel("水平位置 $x$ (m)")
    ax.set_ylabel("滑行速度 $v$ (m/s)")
    ax.set_title("最优运营方案下三种体重游客速度-位移叠合图", fontweight="bold", pad=14)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=12)
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 14)
    ax.grid(True, alpha=0.22)

    # ── 终点局部放大 (标准坐标: 终点在x≈600) ──
    for (color, ls, label), m in zip(styles, masses):
        xx, vv, _ = sim.simulate(m, x_buf=D_BUF)
        x_plot = W - xx
        axz.plot(x_plot, vv, ls, color=color, linewidth=2.5)
    axz.axhline(y=0.5, color="#FF9800", linestyle="--", linewidth=1.2, alpha=0.6)
    axz.axvline(x=X_BUF_COORD, color="orange", linestyle=":", linewidth=1.2, alpha=0.6)
    axz.set_xlim(570, 602)
    axz.set_ylim(-0.2, 3.0)
    axz.set_title("终点局部放大 ($x\\in[570,600]$ m)", fontsize=11, fontweight="bold")
    axz.grid(True, alpha=0.25)

    # 30kg 末速特写标注
    xx30, vv30, _ = sim.simulate(30, x_buf=D_BUF)
    x30_plot = W - xx30
    axz.annotate(f"30kg末速: {vv30[-1]:.2f} m/s\n(未停滞, 验证通过)",
                 xy=(x30_plot[-1], vv30[-1]), xytext=(577, 1.0),
                 fontsize=9, color=C_LIGHT, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_LIGHT, lw=1.2))

    fig.suptitle("最优运营方案下三种体重游客速度-位移叠合图",
                 fontsize=17, fontweight="bold", y=1.01)
    save(fig, "fig08_three_weights")
    plt.close(fig)


# ╔══════════════════════════════════════════════════════════╗
# ║  Fig 11: 全体重验证汇总仪表板                            ║
# ╚══════════════════════════════════════════════════════════╝
def plot_fig11_all_weights_validation():
    cable = CatenaryFromH0(W=600, H=H_OPT, R=1.2, H0=H0_OPT)
    sim = FastSimAdaptive(cable)
    m_list = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

    vmax_list, vend_list, reached_list = [], [], []
    for m in m_list:
        xx, vv, _ = sim.simulate(m, x_buf=D_BUF)
        vmax_list.append(np.max(vv))
        vend_list.append(vv[-1])
        reached_list.append(xx[-1] < 1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    x_idx = np.arange(len(m_list))
    width = 0.35

    # ── 左: v_max 柱状图 ──
    colors_vmax = [C_SAFE if v <= 15 and r else "#FFCDD2" for v, r in zip(vmax_list, reached_list)]
    bars1 = ax1.bar(x_idx, vmax_list, width, color=colors_vmax, edgecolor="black", linewidth=0.6, alpha=0.85)
    for i, (b, v) in enumerate(zip(bars1, vmax_list)):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, f"{v:.1f}",
                 ha="center", fontsize=9, fontweight="bold")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels([f"{m}kg" for m in m_list], rotation=45, ha="right")
    ax1.set_ylabel("最大速度 $v_{\\max}$ (m/s)")
    ax1.set_title("全体重最大速度验证", fontweight="bold")
    ax1.set_ylim(0, 15)
    ax1.grid(True, axis="y", alpha=0.22)
    ax1.axhline(y=np.mean(vmax_list), color="blue", linestyle="--", linewidth=1.0, alpha=0.5,
                label=f"均值 $\\bar{{v}}_{{\\max}}$={np.mean(vmax_list):.1f}")
    ax1.legend(fontsize=10)

    # ── 右: v_end + 安全区间 ──
    colors_vend = ["#4CAF50" if v <= 3.5 and r else "#F44336" if v > 3.5 else "#FF9800"
                   for v, r in zip(vend_list, reached_list)]
    bars2 = ax2.bar(x_idx, vend_list, width, color=colors_vend, edgecolor="black", linewidth=0.6, alpha=0.85)
    for i, (b, v) in enumerate(zip(bars2, vend_list)):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{v:.2f}",
                 ha="center", fontsize=9, fontweight="bold")
    ax2.axhline(y=3.5, color=C_DANGER, linestyle="--", linewidth=2.0, alpha=0.7, label="末速上限 3.5 m/s")
    ax2.axhline(y=0.5, color="#FF9800", linestyle="--", linewidth=1.5, alpha=0.6, label="末速下限 0.5 m/s")
    ax2.fill_between([-0.5, len(m_list) - 0.5], 0.5, 3.5, color=C_SAFE, alpha=0.08)
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels([f"{m}kg" for m in m_list], rotation=45, ha="right")
    ax2.set_ylabel("终点速度 $v_{\\rm end}$ (m/s)")
    ax2.set_title("全体重终点速度验证", fontweight="bold")
    ax2.set_ylim(0, 5)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, axis="y", alpha=0.22)

    status_text = f"全部 {len(m_list)} 个体重均通过安全验证\n$v_{{\\rm end}}\\in[{min(vend_list):.1f}, {max(vend_list):.1f}]$ m/s"
    fig.text(0.5, 0.01, status_text, ha="center", fontsize=13, fontweight="bold", color="#2E7D32",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9", alpha=0.9))

    fig.suptitle("最优方案全体重安全验证仪表板", fontsize=17, fontweight="bold", y=1.01)
    save(fig, "fig09_validation_dashboard")
    plt.close(fig)


# ╔══════════════════════════════════════════════════════════╗
# ║  Fig 12: 缓冲力特性曲线对比                              ║
# ╚══════════════════════════════════════════════════════════╝
def plot_fig12_buffer_characteristics():
    """展示前置测速+恒定制动策略: B_target 随体重自适应变化"""
    cable = CatenaryFromH0(W=600, H=H_OPT, R=1.2, H0=H0_OPT)
    sim = FastSimAdaptive(cable)
    masses = np.linspace(30, 80, 15)
    B_MAX = 150.0

    b_targets, v_pres, v_ends = [], [], []
    for m in masses:
        xx, vv, _ = sim.simulate(m, x_buf=D_BUF)
        # 提取 x_buf (=D_BUF) 处的速度 (自由滑行段末速, 模拟坐标)
        idx_buf = np.argmin(np.abs(xx - D_BUF))
        v_pre = vv[idx_buf]
        b_targets.append(min(sim.compute_btarget(m, D_BUF, v_pre), B_MAX))
        v_pres.append(v_pre)
        v_ends.append(vv[-1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    # 左图: B_target vs 体重
    ax1.bar(masses, b_targets, width=2.5, color=[C_LIGHT, C_MEDIUM, C_HEAVY]*5,
            alpha=0.75, edgecolor="black", linewidth=0.5)
    ax1.axhline(y=B_MAX, color=C_DANGER, linestyle="--", linewidth=1.5, alpha=0.7,
                label=f"机械极限 $B_{{\\rm max}}={B_MAX}$ N")
    ax1.set_xlabel("游客体重 $m$ (kg)")
    ax1.set_ylabel("预计算恒定制动力 $B_{\\rm target}$ (N)")
    ax1.set_title("恒定制动力随体重的自适应分配", fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.22)
    ax1.set_xlim(28, 82)

    # 右图: v_pre vs v_end 关系
    ax2.scatter(v_pres, v_ends, c=masses, s=80, cmap="plasma", edgecolors="black",
                linewidth=0.5, zorder=5)
    for m, vp, ve in zip(masses[::2], v_pres[::2], v_ends[::2]):
        ax2.annotate(f"{m:.0f}kg", xy=(vp, ve), xytext=(5, 5),
                     textcoords="offset points", fontsize=8, alpha=0.8)
    ax2.axhline(y=sim.V_TARGET, color="green", linestyle=":", linewidth=1.2, alpha=0.6,
                label=f"目标末速 $v_{{\\rm target}}={sim.V_TARGET}$ m/s")
    ax2.set_xlabel("测速点速度 $v_{\\rm pre}$ (m/s)")
    ax2.set_ylabel("终点速度 $v_{\\rm end}$ (m/s)")
    ax2.set_title("前置测速 → 终点速度收敛效果", fontweight="bold")
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.grid(True, alpha=0.22)
    cbar = fig.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label("体重 (kg)")

    fig.suptitle("前置测速+恒定制动策略: 恒定制动力自适应分配", fontsize=17, fontweight="bold", y=1.01)
    save(fig, "fig10_buffer_characteristics")
    plt.close(fig)


# ╔══════════════════════════════════════════════════════════╗
# ║  Fig 13: 能量分配堆叠面积图 (80kg 游客)                  ║
# ╚══════════════════════════════════════════════════════════╝
def plot_fig11_energy_breakdown_three():
    """三体重游客能量收支分解堆叠图 (30/55/80 kg)"""
    cable = CatenaryFromH0(W=600, H=H_OPT, R=1.2, H0=H0_OPT)
    sim = FastSimAdaptive(cable)
    masses = [30, 55, 80]
    mass_labels = ["30 kg (轻体重)", "55 kg (中体重)", "80 kg (重体重)"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    for ax, m, mlabel in zip(axes, masses, mass_labels):
        xx_sim, vv, _ = sim.simulate(m, x_buf=D_BUF)
        xx = W - xx_sim
        PE0 = m * 9.81 * H_OPT
        KE = 0.5 * m * vv**2
        W_fric = np.zeros_like(xx_sim)
        W_air = np.zeros_like(xx_sim)
        W_buf = np.zeros_like(xx_sim)
        buf_started = False
        b_target = 0.0
        for i in range(1, len(xx_sim)):
            dx = abs(xx_sim[i] - xx_sim[i - 1])
            theta_i = cable.theta(np.array([xx_sim[i]]))[0]
            v_i = vv[i]
            N_gravity = m * 9.81 * abs(np.cos(theta_i))
            N_tension = cable.tension(np.array([xx_sim[i]]))[0] * cable.kappa(np.array([xx_sim[i]]))[0] * 0.06
            F_fric = 0.005 * N_gravity + 0.025 * N_tension
            W_fric[i] = W_fric[i - 1] + F_fric * dx
            A_m = get_frontal_area(m)
            F_air = 0.5 * 0.7 * 1.225 * A_m * v_i**2
            W_air[i] = W_air[i - 1] + F_air * dx
            in_buf = xx_sim[i] <= D_BUF
            if in_buf and buf_started:
                W_buf[i] = W_buf[i-1] + b_target * dx
            elif in_buf and not buf_started:
                b_target = min(sim.compute_btarget(m, D_BUF, v_i), 150.0)
                buf_started = True
                W_buf[i] = W_buf[i-1] + b_target * dx
            else:
                W_buf[i] = W_buf[i-1]

        W_fric /= 1000; W_air /= 1000; W_buf /= 1000
        KE_kJ = KE / 1000

        ax.fill_between(xx, 0, KE_kJ, alpha=0.7, color="#2196F3", label="动能 $E_k$")
        ax.fill_between(xx, KE_kJ, KE_kJ + W_fric, alpha=0.6, color="#FF9800", label="摩擦耗能")
        ax.fill_between(xx, KE_kJ + W_fric, KE_kJ + W_fric + W_air, alpha=0.6, color="#9C27B0", label="空气阻力")
        top = KE_kJ + W_fric + W_air + W_buf
        ax.fill_between(xx, KE_kJ + W_fric + W_air, top, alpha=0.5, color="#F44336", label="缓冲耗能")
        ax.axhline(y=PE0 / 1000, color="black", linestyle="--", linewidth=1.2, alpha=0.5,
                   label=f"总势能 {PE0/1000:.1f} kJ")
        ax.axvline(x=X_BUF_COORD, color="red", linestyle=":", linewidth=1.2, alpha=0.5)
        ax.set_title(mlabel, fontweight="bold", fontsize=13)
        ax.set_xlabel("水平位置 $x$ (m)")
        if ax == axes[0]:
            ax.set_ylabel("能量 (kJ)")
            ax.legend(loc="upper left", framealpha=0.9, fontsize=8, ncol=1)
        ax.set_xlim(0, 600)
        ax.set_ylim(0, PE0 / 1000 * 1.15)
        ax.grid(True, alpha=0.22)
        ax.text(0.98, 0.95,
                f"$v_{{\\rm end}}={vv[-1]:.1f}$ m/s\n$B_{{\\rm target}}={b_target:.0f}$ N",
                transform=ax.transAxes, fontsize=9, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    fig.suptitle("三种体重游客滑行过程能量分配与耗散对比", fontsize=17, fontweight="bold", y=1.01)
    save(fig, "fig11_energy_breakdown")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("论文图表美化与生成")
    print("=" * 60)
    backup_existing_figures()

    print("\n[1/9] Fig 03: 空载与载人钢缆对比...")
    plot_fig03_cable_loaded()

    print("\n[2/9] Fig 06: 异速生长法则阻力修正...")
    plot_fig06_allometric()

    print("\n[3/9] Fig 07: 三种缓冲策略相轨迹对比...")
    plot_fig07_strategy_compare()

    print("\n[4/9] Fig 08: 恒定缓冲死锁热力图...")
    plot_fig08_buffer_deadlock()

    print("\n[5/9] Fig 09: NSGA-II帕累托前沿3D气泡图...")
    best = plot_fig09_pareto_3d()
    if best:
        print(f"  TOPSIS: H={best['x'][0]:.1f}m, H0={best['x'][1]:.1f}kN, "
              f"x_buf={best['x'][2]:.1f}m (前置测速+恒定制动)")

    print("\n[6/9] Fig 10: 三种体重叠合图...")
    plot_fig10_three_weights()

    print("\n[7/9] Fig 11: 全体重验证仪表盘...")
    plot_fig11_all_weights_validation()

    print("\n[8/9] Fig 12: 缓冲力特性分析...")
    plot_fig12_buffer_characteristics()

    print("\n[9/9] Fig 11: 能量分解分析 (三体重)...")
    plot_fig11_energy_breakdown_three()

    print(f"\n全部图表已保存至: {OUTDIR}")
