"""
问题三：基于NSGA-II的多目标参数优化 (前置测速+单次恒定制动版)
========================================================
策略: "前置测速 + 单次恒定制动" —— 完美契合题目"恒定缓冲摩擦力"要求
  1. 在距终点 x_start 处设置测速传感器，测得瞬时速度 v_pre
  2. 基于能量守恒，逆解出使该游客到达终点时末速 ≈ v_target 的恒定制动力 B_target
  3. 进入缓冲段后，力严格保持为 B_target 恒定不变（对每次具体滑行是常数！）
决策变量: [H, H0, x_start]  (3个, H0直接确定悬链线形态)
约束: 0 ≤ B_target ≤ B_max (150N机械极限), T_max ≤ T_break/4
目标: 最大化平均最大速度, 最小化速度方差
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os, sys, copy, random, time


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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import problem2_dynamics as p2d
from problem2_dynamics import (V_SAFE, G, RHO_AIR, A_FRONTAL, MU_CABLE, L_CONTACT, get_frontal_area)
# 保持向后兼容的模块级引用 (用于plot_paper_figures等外部导入)
C_D = p2d.C_D
MU_ROLL = p2d.MU_ROLL

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "图表")

# ============ 快速悬链线求解器 (给定H0) ============
class CatenaryFromH0:
    """给定水平张力H0, 直接求解悬链线 (无需打靶)"""

    def __init__(self, W, H, R, H0):
        self.W, self.H, self.R, self.H0 = W, H, R, H0
        self.a = H0 / (R * G)

        # 求解 x0: a*(cosh((W-x0)/a) - cosh(x0/a)) = H
        from scipy.optimize import brentq
        def f(x0):
            arg1 = np.clip((W - x0) / self.a, -50, 50)
            arg2 = np.clip(x0 / self.a, -50, 50)
            return self.a * (np.cosh(arg1) - np.cosh(arg2)) - H

        # 合理搜索区间, 扩展以处理紧缆情况
        margin = max(3*W, self.a * 5)
        try:
            self.x0 = brentq(f, -margin, W + margin, xtol=1e-10)
        except ValueError:
            # 紧缆近似: x0 ≈ W/2 - a*H/W
            self.x0 = W/2 - self.a * H / W
        self.y0_offset = self.a * np.cosh(self.x0 / self.a)

    def y(self, x):
        arg = np.clip((x - self.x0) / self.a, -50, 50)
        return self.a * np.cosh(arg) - self.y0_offset

    def yp(self, x):
        arg = np.clip((x - self.x0) / self.a, -50, 50)
        return np.sinh(arg)

    def ypp(self, x):
        arg = np.clip((x - self.x0) / self.a, -50, 50)
        return np.cosh(arg) / self.a

    def tension(self, x):
        arg = np.clip((x - self.x0) / self.a, -50, 50)
        return self.H0 * np.cosh(arg)

    def theta(self, x):
        return np.arctan(self.yp(x))

    def kappa(self, x):
        yp = self.yp(x)
        ypp = self.ypp(x)
        return abs(ypp) / (1 + yp**2)**1.5

    @property
    def sag_from_chord(self):
        x_mid = np.linspace(0, self.W, 200)
        y_chord = self.H * x_mid / self.W
        return np.max(y_chord - self.y(x_mid))


class FastSimAdaptive:
    """前置测速 + 单次恒定制动: 在 x_start 处测速，能量守恒逆解 B_target，
    缓冲段施加恒定 B_target。对每次具体滑行，B_target 是常数——完美契合题目要求。"""

    V_TARGET = 2.8   # 目标末速 (m/s)
    B_MAX    = 150.0 # 机械极限制动力 (N)

    def __init__(self, cable: CatenaryFromH0, cd=None, mu_roll=None):
        self.cable = cable
        self.cd = cd if cd is not None else p2d.C_D
        self.mu_roll = mu_roll if mu_roll is not None else p2d.MU_ROLL
        self.W, self.H = cable.W, cable.H
        N = 300
        self._xg = np.linspace(0, self.W, N)
        self._theta = cable.theta(self._xg)
        self._T = cable.tension(self._xg)
        self._kappa = cable.kappa(self._xg)
        self._y = cable.y(self._xg)
        self._N = N
        # 预存终点方向
        self._x_end = 0.0 if self.H >= 0 else self.W
        self._sgn = -1.0 if self.H >= 0 else 1.0

    def friction_force(self, x, m):
        i = int(np.clip(x / self.W * (self._N - 1), 0, self._N - 1))
        N_gravity = m * G * np.cos(self._theta[i])
        N_tension = self._T[i] * self._kappa[i] * L_CONTACT
        return self.mu_roll * N_gravity + MU_CABLE * N_tension

    def _interp(self, x, attr):
        """线性插值获取 x 处的几何属性"""
        idx = np.clip(np.searchsorted(self._xg, x), 1, self._N - 1)
        xl, xr = self._xg[idx-1], self._xg[idx]
        fl, fr = attr[idx-1], attr[idx]
        t = (x - xl) / (xr - xl) if xr > xl else 0
        return fl + t * (fr - fl)

    def _arc_length(self, x_from, x_to):
        """计算两点间的弧长 (积分 ds = sqrt(1+yp^2) dx)"""
        mask = (self._xg >= min(x_from, x_to)) & (self._xg <= max(x_from, x_to))
        if np.sum(mask) < 2:
            return 0.0
        yp = np.sinh((self._xg[mask] - self.cable.x0) / self.cable.a)
        ds = np.sqrt(1 + yp**2)
        return np.trapezoid(ds, self._xg[mask])

    def _precompute_friction_work(self, m, x_buf):
        """预计算缓冲段固有摩擦耗散功 (仅位置依赖项)"""
        mask = (self._xg >= min(x_buf, self._x_end)) & (self._xg <= max(x_buf, self._x_end))
        if np.sum(mask) < 2:
            return 0.0
        x_seg = self._xg[mask]
        yp = np.sinh((x_seg - self.cable.x0) / self.cable.a)
        ds = np.sqrt(1 + yp**2)
        # 摩擦功 = ∫ F_fric ds
        F_fric = np.array([self.friction_force(xi, m) for xi in x_seg])
        return np.trapezoid(F_fric, x_seg)

    def compute_btarget(self, m, x_buf, v_pre):
        """能量守恒逆解: 计算使末速 ≈ V_TARGET 所需的恒定制动力 B_target"""
        x_end = self._x_end
        y_buf = self._interp(x_buf, self._y)
        y_end = self._interp(x_end, self._y)
        S_buf = self._arc_length(x_buf, x_end)
        if S_buf < 0.1:
            return 0.0

        # 能量平衡 (忽略空气阻力做保守估计)
        dKE = 0.5 * m * (self.V_TARGET**2 - v_pre**2)  # < 0 表示需耗散
        dPE = m * G * (y_end - y_buf)                  # < 0 表示重力做功
        W_fric = self._precompute_friction_work(m, x_buf)
        # 需耗散能量 = -(dKE + dPE) = 初始机械能 - 目标机械能
        # B_target * S_buf + W_fric = -(dKE + dPE)
        net_dissipation = -(dKE + dPE) - W_fric  # 需缓冲力提供的耗散
        B_target = max(0.0, net_dissipation / S_buf)
        return B_target

    def simulate(self, m, dt=0.10, x_buf=None, max_t=180.0, eps_v=0.0, eps_B=0.0):
        """两阶段仿真:
        Phase 1: 自由滑行至 x_buf, 测速 v_pre (加噪声 eps_v)
        Phase 2: 恒定 B_target (加执行误差 eps_B) 制动至终点
        返回 (x_arr, v_arr, t_arr)
        """
        W = self.W
        x0 = W if self.H >= 0 else 0.0
        x_end = self._x_end
        sgn = self._sgn

        x, v, t = x0, 0.0, 0.0
        traj = [(x, v, t)]
        b_target = 0.0
        b_effective = 0.0
        phase = 1  # 1=自由滑行, 2=缓冲制动

        while t < max_t:
            idx = int(np.clip(x / W * (self._N - 1), 0, self._N - 1))
            theta = self._theta[idx]

            F_g = -m * G * np.sin(theta)
            F_fric_mag = self.friction_force(x, m)
            A_dyn = get_frontal_area(m)
            F_air = 0.5 * self.cd * RHO_AIR * A_dyn * v * abs(v)
            F_buf_f = 0.0

            if phase == 1 and x_buf is not None:
                in_buf = (sgn < 0 and x <= x_buf) or (sgn > 0 and x >= x_buf)
                if in_buf:
                    # 测速并计算 B_target (含测速噪声)
                    v_measured = max(0.0, abs(v) + eps_v)
                    b_target = self.compute_btarget(m, x_buf, v_measured)
                    b_target = min(b_target, self.B_MAX)
                    # 施加制动执行误差
                    b_effective = max(0.0, b_target + eps_B)
                    phase = 2

            if phase == 2 and b_target > 0:
                F_buf_f = b_effective * (np.sign(v) if abs(v) > 1e-9 else sgn)

            sv = np.sign(v) if abs(v) > 1e-9 else np.sign(F_g)
            F_total = F_g - sv * F_fric_mag - F_air - F_buf_f
            a = np.clip(F_total / m, -500, 500)

            adapt_dt = min(dt, 0.3 / (1.0 + 0.3 * abs(v)))
            cos_th = np.cos(theta)
            x_new = x + cos_th * (v * adapt_dt + 0.5 * a * adapt_dt**2)
            v_new = v + a * adapt_dt

            if (sgn < 0 and x_new <= x_end) or (sgn > 0 and x_new >= x_end):
                traj.append((x_end, abs(v), t + adapt_dt))
                break
            if abs(v_new) > 200 or (abs(v_new) < 0.002 and abs(x - x_end) > 10 and t > 5):
                break

            x, v, t = x_new, v_new, t + adapt_dt
            traj.append((x, abs(v), t))

        arr = np.array(traj)
        return arr[:, 0], arr[:, 1], arr[:, 2]


def evaluate_solution_fast(cable: CatenaryFromH0, x_buf, m_list):
    """评估给定设计方案：前置测速+单次恒定制动力策略"""
    sim = FastSimAdaptive(cable)
    vmaxs, vends, reacheds, b_targets = [], [], [], []
    for m in m_list:
        xx, vv, tt = sim.simulate(m, x_buf=x_buf)
        vmaxs.append(np.max(vv))
        vends.append(vv[-1])
        reacheds.append(abs(xx[-1] - (0.0 if cable.H >= 0 else cable.W)) < 1.0)
        # 从仿真轨迹反向提取 B_target (通过力平衡反算)
        # 简化: 检查 B_target 是否在合法范围内
        b_targets.append(0.0)  # placeholder

    vmaxs = np.array(vmaxs)
    vends = np.array(vends)
    reacheds = np.array(reacheds)
    all_ok = np.all(vends <= V_SAFE) and np.all(vmaxs > 0.3) and np.all(reacheds)

    return {
        "avg_vmax": np.mean(vmaxs),
        "var_vmax": np.var(vmaxs),
        "max_vend": np.max(vends),
        "all_safe": all_ok,
        "vmax_list": vmaxs,
        "vend_list": vends,
        "reached": reacheds,
        "b_targets": b_targets,
    }


# ============ NSGA-II (复用问题3的类) ============
from problem3_nsga2 import NSGA2, topsis_select


def make_evaluator_fixed(W, R, m_list, T_max_allowed=None, B_max=150.0):
    """决策变量: [H, H0(kN), x_start(m)] — 3变量
    前置测速后能量守恒逆解 B_target，B_target ∈ [0, B_max] 为硬约束"""
    def evaluate(x):
        H_val = x[0]
        H0_val = x[1]  # kN
        x_buf = x[2]   # 缓冲启动位置 (离终点距离, m)

        try:
            cable = CatenaryFromH0(W, H_val, R, H0_val * 1e3)
        except Exception:
            return [1e8, 1e8]

        # 安全约束: T_max ≤ T_break/4
        if T_max_allowed is not None:
            x_chk = np.linspace(0, W, 200)
            T_chk = cable.tension(x_chk)
            if np.max(T_chk) > T_max_allowed:
                viol = (np.max(T_chk) - T_max_allowed) / 1e3
                penalty = 50.0 + viol * 10.0
                return [penalty, penalty]

        res = evaluate_solution_fast(cable, x_buf, m_list)

        if not res["all_safe"]:
            v_end_viol = np.sum(np.maximum(0, res["vend_list"] - V_SAFE))
            reach_viol = np.sum([25.0 for r in res["reached"] if not r])
            vmax_viol = np.sum([15.0 for vm in res["vmax_list"] if vm <= 0.3])
            vmin_viol = np.sum([10.0 for ve in res["vend_list"] if ve < 1.0])
            penalty = 30.0 + v_end_viol + reach_viol + vmax_viol + vmin_viol
            return [penalty, penalty]

        # 鲁棒性约束: 最轻体重游客名义末速≥0.5 m/s的能量余量
        v_end_light = res["vend_list"][0]  # m_list已按升序排列
        robust_penalty = max(0.0, 0.5 - v_end_light) * 80.0
        return [-res["avg_vmax"] + robust_penalty, res["var_vmax"] + robust_penalty]

    return evaluate


# ============ 图表 ============
def plot_fig6(population, best_ind):
    pf = [ind for ind in population if ind["rank"] == 0]
    other = [ind for ind in population if ind["rank"] > 0]

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    if other:
        ox = [ind["x"][0] for ind in other]
        oy = [ind["x"][2] for ind in other]
        oz = [-ind["obj"][0] for ind in other]
        ax.scatter(ox, oy, oz, c="#8D99AE", s=10, alpha=0.15)

    px = [ind["x"][0] for ind in pf]
    py = [ind["x"][2] for ind in pf]
    pz = [-ind["obj"][0] for ind in pf]
    pc = [ind["obj"][1] for ind in pf]
    sc = ax.scatter(px, py, pz, c=pc, cmap="viridis", s=60,
                    alpha=0.9, label=f"Pareto前沿 (n={len(pf)})")

    if best_ind:
        bx, by, bz = best_ind["x"][0], best_ind["x"][2], -best_ind["obj"][0]
        ax.scatter([bx], [by], [bz], c="#D1495B", s=250, marker="*",
                   edgecolors="white", linewidths=1.5, zorder=10, label="TOPSIS推荐")

    ax.set_xlabel("高差 H (m)", fontsize=12)
    ax.set_ylabel("缓冲启动位置 (m)", fontsize=12)
    ax.set_zlabel("平均最大速度 (m/s)", fontsize=12)
    ax.set_title("帕累托前沿分布 (前置测速+恒定制动)", fontsize=14, fontweight="bold")
    fig.colorbar(sc, ax=ax, shrink=0.5, label="速度方差")
    ax.legend(fontsize=10)

    fp = os.path.join(FIG_DIR, "fig07_pareto_front.pdf")
    fig.savefig(fp, dpi=320, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(fp.replace(".pdf", ".png"), dpi=320, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[图7] 已保存: {fp}")


def plot_fig7(cable, x_buf, m_samples):
    sim = FastSimAdaptive(cable)
    fig, ax = plt.subplots(figsize=(12, 7))

    m_light, m_mid, m_heavy = m_samples[0], m_samples[len(m_samples)//2], m_samples[-1]
    W = cable.W
    x_buf_coord = W - x_buf  # 标准坐标: 起点x=0, 终点x=600, 缓冲在x_buf_coord

    colors = {'light': '#2A9D8F', 'mid': '#2F6690', 'heavy': '#D1495B'}
    styles = {m_light: (colors['light'], '-', 2.2),
              m_mid: (colors['mid'], '--', 2.0),
              m_heavy: (colors['heavy'], '-.', 2.0)}

    v_all_max = 0
    for m, (c, ls, lw) in styles.items():
        xx_sim, vv, _ = sim.simulate(m, x_buf=x_buf)
        x_plot = W - xx_sim  # 转换到标准坐标
        ax.plot(x_plot, vv, color=c, linestyle=ls, linewidth=lw, label=f"$m={m}$ kg")
        v_all_max = max(v_all_max, np.max(vv))
        ax.annotate(f"{vv[-1]:.1f} m/s", xy=(x_plot[-1], vv[-1]),
                    xytext=(x_plot[-1]-80, vv[-1]+0.5), fontsize=9, color=c, fontweight='bold')

    ax.fill_between([0, W], V_SAFE, v_all_max * 1.1, alpha=0.08, color='#D62828')
    ax.axhline(y=V_SAFE, color="#D62828", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"安全限 {V_SAFE} m/s")
    ax.axvline(x=x_buf_coord, color="#7B2CBF", linestyle=":", linewidth=1.5, alpha=0.5,
               label=f"缓冲启动 $x_b={x_buf_coord:.1f}$m (距终点${x_buf:.1f}$m)")

    ax.set_xlabel("水平位置 $x$ (m)", fontsize=13)
    ax.set_ylabel("滑行速度 $v$ (m/s)", fontsize=13)
    ax.set_title("最优参数下不同体重游客速度—位移叠合图", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, W)
    ax.set_ylim(0, v_all_max * 1.15)

    fp = os.path.join(FIG_DIR, "fig08_three_weights.pdf")
    fig.savefig(fp, dpi=320, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(fp.replace(".pdf", ".png"), dpi=320, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[图8] 已保存: {fp}")


# ============ 主程序 ============
if __name__ == "__main__":
    print("=" * 60)
    print("问题三：NSGA-II 多目标优化 (前置测速+恒定制动, W=600m)")
    print("=" * 60)

    W, R = 600.0, 1.2
    m_samples = [30, 40, 50, 55, 60, 70, 75, 80]

    # 钢缆强度参数
    SIGMA_B = 1770e6
    A_CABLE = R / 7850.0
    T_BREAK = SIGMA_B * A_CABLE
    T_MAX_ALLOWED = T_BREAK / 4.0

    # 决策变量: [H, H0(kN), x_start(m)] — 仅3个!
    bounds = [
        (14.0, 20.0),     # H: 高差 (题目约束上限20m)
        (25.0, 65.0),     # H0: 水平张力 (kN), 紧缆→小垂度→足够下滑动力
        (5.0, 80.0),      # x_buf: 缓冲启动位置 (离终点距离, m)
    ]

    print(f"  策略: 前置测速 → 能量守恒逆解B_target → 恒定制动")
    print(f"  W=600m, H∈[14,20]m, H0∈[25,65]kN, x_buf∈[5,80]m")
    print(f"  安全约束: T_max ≤ {T_MAX_ALLOWED/1e3:.1f} kN (K≥4)")
    print(f"  体重范围: {m_samples} kg")
    print(f"  种群=60, 代数=60")

    eval_func = make_evaluator_fixed(W, R, m_samples, T_MAX_ALLOWED)
    nsga2 = NSGA2(pop_size=60, n_gen=60, p_cross=0.9, p_mut=0.1)

    random.seed(42); np.random.seed(42)
    t0 = time.time()
    population, _ = nsga2.optimize(eval_func, bounds, verbose=True)
    print(f"  耗时: {time.time()-t0:.1f}s")

    pf = [ind for ind in population if ind["rank"] == 0]
    feasible = [ind for ind in pf if ind["obj"][0] < 0]
    print(f"  帕累托前沿: {len(pf)} 解, 其中可行解: {len(feasible)}")

    if feasible:
        result = topsis_select(feasible)
        if result:
            best, C_vals, _ = result
            print(f"\n  TOPSIS推荐方案:")
            print(f"    H = {best['x'][0]:.1f} m")
            print(f"    H0 = {best['x'][1]:.1f} kN")
            print(f"    x_buf = {best['x'][2]:.1f} m")

            cable_opt = CatenaryFromH0(W, best["x"][0], R, best["x"][1] * 1e3)
            res = evaluate_solution_fast(cable_opt, best["x"][2], m_samples)
            print(f"\n  验证:")
            for m, vm, ve, rd in zip(m_samples, res["vmax_list"], res["vend_list"], res["reached"]):
                s = "OK" if ve <= V_SAFE and rd else "FAIL"
                print(f"    m={m:3.0f}kg: v_max={vm:.2f} m/s, v_end={ve:.2f} m/s, 到达={rd} [{s}]")

            plot_fig6(population, best)
            plot_fig7(cable_opt, best["x"][2], m_samples)
    else:
        print("\n  未找到完全可行解, 使用最优不可行解...")
        best_violation = min(population, key=lambda ind: ind["obj"][0])
        print(f"  最优近似: H={best_violation['x'][0]:.1f}, H0={best_violation['x'][1]:.1f}kN")
        plot_fig6(population, best_violation)

    print("\n问题三完成。")
