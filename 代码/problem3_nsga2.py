"""
问题三：基于NSGA-II的多目标参数优化
=====================================
决策变量: [H, L_cable, x_start_buf, F_buf]
目标1: 最大化平均最大速度 (满意度)
目标2: 最小化最大速度方差 (群体一致性)
约束: v_end ≤ 3.5m/s, v>0 全程, H∈[5,20]
方法: NSGA-II + TOPSIS筛选
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
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
from problem2_dynamics import DynamicsSimulator, V_SAFE, G, RHO_AIR, C_D, A_FRONTAL, MU_ROLL, FIG_DIR

# ============ 抛物线近似钢缆模型 (快速计算) ============
class ParabolicCable:
    """抛物线近似钢缆 — 适用于小垂度悬索 (sag/span < 1/8)"""

    def __init__(self, W: float, H: float, R: float, L: float = None, H0: float = None):
        self.W, self.H, self.R = W, H, R
        if H0 is not None:
            self.H0 = H0
            self.sag = R * G * W**2 / (8 * H0)
        elif L is not None:
            chord = np.sqrt(W**2 + H**2)
            L_extra = L - chord
            if L_extra <= 0:
                self.sag = 0.01  # 防止无解
            else:
                self.sag = np.sqrt(3 * W * max(L_extra - H**2/(2*W), 0.001) / 8)
            self.H0 = R * G * W**2 / (8 * self.sag)
        else:
            raise ValueError("Need L or H0")

        self.L_est = np.sqrt(W**2 + H**2) + H**2/(2*W) + 8*self.sag**2/(3*W)

    def y(self, x):
        return self.H * x / self.W + 4 * self.sag * (x / self.W) * (1 - x / self.W)

    def yp(self, x):
        return self.H / self.W + 4 * self.sag * (1 - 2 * x / self.W) / self.W

    def ypp(self, x=None):
        return -8 * self.sag / self.W**2

    def theta(self, x):
        return np.arctan(self.yp(x))

    def tension(self, x):
        yp = self.yp(x)
        return self.H0 * np.sqrt(1 + yp**2)

    def kappa(self, x):
        ypp = self.ypp()
        yp = self.yp(x)
        return abs(ypp) / (1 + yp**2)**1.5


class FastSimulator:
    """基于抛物线钢缆的快速动力学仿真器"""

    def __init__(self, cable: ParabolicCable):
        self.cable = cable
        self.W = cable.W
        self.H = cable.H
        self._precompute()

    def _precompute(self):
        W = self.W
        self._x_grid = np.linspace(0, W, 1000)
        pc = self.cable
        self._yp = pc.yp(self._x_grid)
        self._T  = pc.tension(self._x_grid)
        self._theta = np.arctan(self._yp)
        self._kappa = pc.kappa(self._x_grid)

    def friction_force(self, x, m):
        i = np.clip(int(x / self.W * 999), 0, 999)
        theta = self._theta[i]
        return MU_ROLL * m * G * np.cos(theta)

    def simulate(self, m, dt=0.1, x_buf=None, F_buf=0.0, max_t=200.0):
        W = self.W
        x0 = W if self.H >= 0 else 0.0  # 从高端出发
        x_end = 0.0 if x0 > W/2 else W
        dir_sign = -1.0 if x0 > W/2 else 1.0

        x, v, t = x0, 0.0, 0.0
        traj_x, traj_v, traj_t = [x], [v], [t]

        while t < max_t:
            # 查表获取局部参数
            idx = np.clip(int(x / W * 999), 0, 999)
            theta = self._theta[idx]

            # 力计算
            F_g = -m * G * np.sin(theta)
            F_fric_mag = self.friction_force(x, m)
            F_air = 0.5 * C_D * RHO_AIR * A_FRONTAL * v * abs(v)
            F_buf_f = F_buf * np.sign(v) if (x_buf and x >= x_buf and v != 0) else 0.0
            sgn = np.sign(v) if abs(v) > 1e-9 else np.sign(F_g)
            F_total = F_g - sgn * F_fric_mag - F_air - F_buf_f
            a = np.clip(F_total / m, -200, 200)

            # Euler半步 (快速求解用, 已够精确)
            adapt_dt = min(dt, 0.3 / (1.0 + 0.2 * abs(v)))
            cos_th = np.cos(theta)
            # dx/dt = v·cos(θ): 水平位移与切向速度的几何关系
            x_new = x + cos_th * (v * adapt_dt + 0.5 * a * adapt_dt**2)
            v_new = v + a * adapt_dt
            t_new = t + adapt_dt

            # 越界检测
            if dir_sign < 0 and x_new <= x_end:
                x_new, v_new = x_end, v
                traj_x.append(x_new); traj_v.append(abs(v_new)); traj_t.append(t_new)
                break
            elif dir_sign > 0 and x_new >= x_end:
                x_new, v_new = x_end, v
                traj_x.append(x_new); traj_v.append(abs(v_new)); traj_t.append(t_new)
                break

            if abs(v_new) > 150 or (abs(v_new) < 0.003 and t > 3 and abs(x - x_end) > 5):
                break

            x, v, t = x_new, v_new, t_new
            traj_x.append(x); traj_v.append(abs(v)); traj_t.append(t)

        return np.array(traj_x), np.array(traj_v), np.array(traj_t)


def evaluate_solution(cable: ParabolicCable, x_buf: float, F_buf: float,
                      m_list: list) -> dict:
    """评估一组设计参数的效能"""
    sim = FastSimulator(cable)
    v_max_list, v_end_list, reached_list = [], [], []

    for m in m_list:
        x_arr, v_arr, t_arr = sim.simulate(m, x_buf=x_buf, F_buf=F_buf)
        v_max_list.append(np.max(v_arr))
        v_end_list.append(v_arr[-1])
        reached = abs(x_arr[-1] - (0.0 if cable.H >= 0 else cable.W)) < 0.5
        reached_list.append(reached)

    v_max_arr = np.array(v_max_list)
    v_end_arr = np.array(v_end_list)
    reached_arr = np.array(reached_list)

    all_safe = np.all(v_end_arr <= V_SAFE) and np.all(v_max_arr > 0.5) and np.all(reached_arr)

    return {
        "avg_vmax": np.mean(v_max_arr),
        "var_vmax": np.var(v_max_arr),
        "max_vend": np.max(v_end_arr),
        "all_safe": all_safe,
        "vmax_list": v_max_list,
        "vend_list": v_end_list
    }


# ============ NSGA-II ============
class NSGA2:
    """带精英策略的非支配排序遗传算法 (Deb et al., 2002)"""

    def __init__(self, pop_size=60, n_gen=80, p_cross=0.9, p_mut=0.1,
                 eta_c=15, eta_m=20):
        self.N = pop_size
        self.G = n_gen
        self.pc = p_cross
        self.pm = p_mut
        self.eta_c = eta_c  # SBX分布指数
        self.eta_m = eta_m  # 多项式变异分布指数

    def _dominates(self, a, b):
        """a支配b? (最小化问题)"""
        better = False
        for fa, fb in zip(a["obj"], b["obj"]):
            if fa > fb:
                return False
            if fa < fb:
                better = True
        return better

    def _fast_non_dominated_sort(self, pop):
        F = [[]]  # 帕累托层级列表
        n = len(pop)
        S = [[] for _ in range(n)]
        n_p = [0] * n
        rank = [0] * n

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(pop[i], pop[j]):
                    S[i].append(j)
                elif self._dominates(pop[j], pop[i]):
                    n_p[i] += 1
            if n_p[i] == 0:
                rank[i] = 0
                F[0].append(i)

        k = 0
        while len(F[k]) > 0:
            Q = []
            for i in F[k]:
                for j in S[i]:
                    n_p[j] -= 1
                    if n_p[j] == 0:
                        rank[j] = k + 1
                        Q.append(j)
            k += 1
            F.append(Q)
        F.pop()  # 移除最后的空列表
        return F, rank

    def _crowding_distance(self, pop, front):
        if len(front) <= 2:
            return [1e10] * len(front)

        dist = [0.0] * len(front)
        n_obj = len(pop[0]["obj"])

        for m in range(n_obj):
            idx_sorted = sorted(front, key=lambda i: pop[i]["obj"][m])
            obj_min = pop[idx_sorted[0]]["obj"][m]
            obj_max = pop[idx_sorted[-1]]["obj"][m]
            if obj_max == obj_min:
                continue
            dist[front.index(idx_sorted[0])] = 1e10
            dist[front.index(idx_sorted[-1])] = 1e10
            for k in range(1, len(front) - 1):
                dist[front.index(idx_sorted[k])] += (
                    pop[idx_sorted[k+1]]["obj"][m] - pop[idx_sorted[k-1]]["obj"][m]
                ) / (obj_max - obj_min)
        return dist

    def _tournament_select(self, pop, rank, crowd_dist):
        n = len(pop)
        a, b = random.randint(0, n-1), random.randint(0, n-1)
        if rank[a] < rank[b]:
            return copy.deepcopy(pop[a])
        elif rank[a] > rank[b]:
            return copy.deepcopy(pop[b])
        return copy.deepcopy(pop[a] if crowd_dist[a] > crowd_dist[b] else pop[b])

    def _sbx_crossover(self, p1, p2, bounds):
        c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
        if random.random() > self.pc:
            return c1, c2

        for i in range(len(p1["x"])):
            if random.random() < 0.5:
                u = random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (self.eta_c + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1))
                x1, x2 = p1["x"][i], p2["x"][i]
                lo, hi = bounds[i]
                c1["x"][i] = np.clip(0.5 * ((1 + beta) * x1 + (1 - beta) * x2), lo, hi)
                c2["x"][i] = np.clip(0.5 * ((1 - beta) * x1 + (1 + beta) * x2), lo, hi)
        return c1, c2

    def _poly_mutation(self, ind, bounds):
        for i in range(len(ind["x"])):
            if random.random() > self.pm:
                continue
            u = random.random()
            lo, hi = bounds[i]
            delta = min(ind["x"][i] - lo, hi - ind["x"][i]) / (hi - lo)
            if u <= 0.5:
                delta_q = (2 * u + (1 - 2 * u) * (1 - delta)**(self.eta_m + 1))**(1/(self.eta_m + 1)) - 1
            else:
                delta_q = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta)**(self.eta_m + 1))**(1/(self.eta_m + 1))
            ind["x"][i] = np.clip(ind["x"][i] + delta_q * (hi - lo), lo, hi)

    def optimize(self, eval_func, bounds, n_obj=2, verbose=True):
        """主优化循环"""
        n_var = len(bounds)

        # 初始化种群
        pop = []
        for _ in range(self.N):
            x = [random.uniform(b[0], b[1]) for b in bounds]
            obj = eval_func(x)
            pop.append({"x": x, "obj": obj, "rank": 0, "crowd": 0.0})

        pareto_history = []

        for gen in range(self.G):
            # 非支配排序 + 拥挤度
            fronts, ranks = self._fast_non_dominated_sort(pop)
            for i, front in enumerate(fronts):
                for idx in front:
                    pop[idx]["rank"] = i
            for front in fronts:
                cd = self._crowding_distance(pop, front)
                for j, idx in enumerate(front):
                    pop[idx]["crowd"] = cd[j]

            # 生成子代
            offspring = []
            while len(offspring) < self.N:
                p1 = self._tournament_select(pop, ranks,
                    [pop[i]["crowd"] for i in range(len(pop))])
                p2 = self._tournament_select(pop, ranks,
                    [pop[i]["crowd"] for i in range(len(pop))])
                c1, c2 = self._sbx_crossover(p1, p2, bounds)
                self._poly_mutation(c1, bounds)
                self._poly_mutation(c2, bounds)
                c1["obj"] = eval_func(c1["x"])
                c2["obj"] = eval_func(c2["x"])
                offspring.extend([c1, c2])

            # 合并 + 环境选择
            combined = pop + offspring[:self.N]
            fronts_c, ranks_c = self._fast_non_dominated_sort(combined)
            for i, front in enumerate(fronts_c):
                for idx in front:
                    combined[idx]["rank"] = i

            new_pop = []
            for front in fronts_c:
                if len(new_pop) + len(front) <= self.N:
                    new_pop.extend(front)
                else:
                    cd = self._crowding_distance(combined, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    new_pop.extend(sorted_front[:self.N - len(new_pop)])
                    break

            pop = [combined[i] for i in new_pop]

            # 记录帕累托前沿
            pf = [ind for ind in pop if ind["rank"] == 0]
            pareto_history.append(len(pf))

            if verbose and (gen + 1) % 20 == 0:
                objs = [ind["obj"] for ind in pf]
                avg_f1 = np.mean([o[0] for o in objs])
                avg_f2 = np.mean([o[1] for o in objs])
                print(f"  Gen {gen+1:3d}: Pareto={len(pf):3d}, avg f1={avg_f1:.2f}, avg f2={avg_f2:.4f}")

        return pop, pareto_history


# ============ TOPSIS 筛选 ============
def topsis_select(pareto_front):
    """从帕累托前沿中用TOPSIS+熵权法筛选最终方案"""
    n = len(pareto_front)
    if n == 0:
        return None

    m_obj = np.array([ind["obj"] for ind in pareto_front])  # (n, 2)

    # 归一化
    m_norm = m_obj / np.sqrt(np.sum(m_obj**2, axis=0))

    # 熵权法
    p = m_norm / np.sum(m_norm, axis=0)
    p = np.clip(p, 1e-10, 1)
    e = -np.sum(p * np.log(p), axis=0) / np.log(n)
    w = (1 - e) / np.sum(1 - e)

    # 加权矩阵
    m_weighted = m_norm * w

    # 理想解 / 反理想解 (两个目标都是最小化)
    ideal_pos = np.min(m_weighted, axis=0)
    ideal_neg = np.max(m_weighted, axis=0)

    # 距离
    d_pos = np.sqrt(np.sum((m_weighted - ideal_pos)**2, axis=1))
    d_neg = np.sqrt(np.sum((m_weighted - ideal_neg)**2, axis=1))

    # 贴近度
    C = d_neg / (d_pos + d_neg)
    best_idx = np.argmax(C)

    return pareto_front[best_idx], C, best_idx


# ============ 图6: 帕累托前沿 ============
def plot_fig6_pareto(population, best_ind=None):
    """图6: 帕累托前沿3D散点图"""
    pf = [ind for ind in population if ind["rank"] == 0]
    other = [ind for ind in population if ind["rank"] > 0]

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    # 非帕累托解 (灰色)
    if other:
        ox = [ind["x"][0] for ind in other]
        oy = [ind["x"][2] for ind in other]
        oz1 = [ind["obj"][0] for ind in other]
        oz2 = [ind["obj"][1] for ind in other]
        ax.scatter(ox, oy, oz1, c="#8D99AE", s=15, alpha=0.2, label="其他解")

    # 帕累托前沿
    px = [ind["x"][0] for ind in pf]
    py = [ind["x"][2] for ind in pf]
    pz1 = [ind["obj"][0] for ind in pf]
    pz2 = [ind["obj"][1] for ind in pf]
    colors_pf = pz2  # 用f2着色
    sc = ax.scatter(px, py, pz1, c=colors_pf, cmap="plasma", s=50, alpha=0.9,
                    label=f"帕累托前沿 (n={len(pf)})")

    # 最优解
    if best_ind:
        bx, by, bz1 = best_ind["x"][0], best_ind["x"][2], best_ind["obj"][0]
        ax.scatter([bx], [by], [bz1], c="#D1495B", s=200, marker="*",
                   edgecolors="white", linewidths=1.5, zorder=10, label="TOPSIS推荐")

    ax.set_xlabel("高差 H (m)", fontsize=12)
    ax.set_ylabel("缓冲位置 $x_{buf}$ (m)", fontsize=12)
    ax.set_zlabel("目标1: 平均最大速度 (m/s)", fontsize=12)
    ax.set_title("帕累托前沿分布", fontsize=14, fontweight="bold")
    fig.colorbar(sc, ax=ax, shrink=0.5, aspect=20, label="目标2: 速度方差")
    ax.legend(fontsize=10)

    path = os.path.join(FIG_DIR, "fig06_pareto_front.pdf")
    fig.savefig(path, dpi=320, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(path.replace(".pdf", ".png"), dpi=320, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[图6] 已保存: {path}")
    return path


# ============ 图7: 最优参数速度叠合图 ============
def plot_fig7_three_weights(cable: ParabolicCable, x_buf: float, F_buf: float):
    """图7: 最优参数下30/55/80kg三人速度-位移叠合图"""
    sim = FastSimulator(cable)
    fig, ax = plt.subplots(figsize=(10, 6))

    styles = {30: ("#2196F3", "-", 2.0),
              55: ("#4CAF50", "--", 1.8),
              80: ("#FF5722", "-.", 1.8)}

    for m, (color, ls, lw) in styles.items():
        x_arr, v_arr, _ = sim.simulate(m, x_buf=x_buf, F_buf=F_buf)
        ax.plot(x_arr, v_arr, color=color, linestyle=ls, linewidth=lw,
                label=f"m = {m} kg")

    # 安全线 + 缓冲标记
    ax.axhline(y=V_SAFE, color="red", linestyle=":", linewidth=1.5, alpha=0.6,
               label=f"安全限 {V_SAFE} m/s")
    ax.axvline(x=x_buf, color="purple", linestyle="--", linewidth=1.2, alpha=0.5,
               label=f"缓冲启动 x={x_buf:.0f}m")

    ax.set_xlabel("水平位移 x (m)", fontsize=13)
    ax.set_ylabel("滑行速度 v (m/s)", fontsize=13)
    ax.set_title("最优参数下不同体重游客速度—位移叠合图", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, cable.W)

    path = os.path.join(FIG_DIR, "fig07_three_weights.pdf")
    fig.savefig(path, dpi=320, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(path.replace(".pdf", ".png"), dpi=320, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[图7] 已保存: {path}")
    return path


# ============ 适应度评估函数 ============
def make_evaluator(W, R, m_list):
    """构造NSGA-II的评估函数 (closure封装固定参数)"""
    chord = np.sqrt(W**2 + 25)  # H≈5时弦长近似

    def evaluate(x):
        H_val = x[0]        # 高差 5-20 m
        L_val = x[1]        # 钢缆总长 > chord
        x_buf = x[2]        # 缓冲启动位置
        F_buf = x[3]        # 缓冲力

        try:
            cable = ParabolicCable(W, H_val, R, L=L_val)
        except:
            return [1e6, 1e6]  # 无效解

        res = evaluate_solution(cable, x_buf, F_buf, m_list)

        if not res["all_safe"]:
            # 惩罚不可行解
            penalty = 1e3 + np.mean([max(0, ve - V_SAFE) for ve in res["vend_list"]])
            return [penalty, 1e3]

        # 目标1: 负平均最大速度 (因为NSGA-II默认最小化)
        f1 = -res["avg_vmax"]
        # 目标2: 速度方差
        f2 = res["var_vmax"]

        return [f1, f2]

    return evaluate


# ============ 主程序 ============
if __name__ == "__main__":
    print("=" * 60)
    print("问题三：NSGA-II 多目标参数优化")
    print("=" * 60)

    W, R = 600.0, 1.2
    m_samples = [30, 40, 50, 55, 60, 70, 75, 80]

    # 决策变量边界: [H, L, x_start_buf, F_buf]
    chord_min = np.sqrt(W**2 + 5**2)
    chord_max = np.sqrt(W**2 + 20**2)
    bounds = [
        (5.0, 20.0),               # H: 高差
        (chord_min + 0.3, chord_max + 8.0),  # L: 钢缆总长
        (200.0, 560.0),             # x_start_buf
        (10.0, 180.0)               # F_buf
    ]

    print(f"  决策变量边界: H∈[5,20], L∈[{bounds[1][0]:.1f},{bounds[1][1]:.1f}], "
          f"x_buf∈[200,560], F_buf∈[10,180]")
    print(f"  采样体重: {m_samples}")
    print(f"  种群={40}, 代数={60}")

    eval_func = make_evaluator(W, R, m_samples)
    nsga2 = NSGA2(pop_size=40, n_gen=60, p_cross=0.9, p_mut=0.1)
    random.seed(42)
    np.random.seed(42)

    t0 = time.time()
    population, pareto_hist = nsga2.optimize(eval_func, bounds, verbose=True)
    t1 = time.time()
    print(f"  优化耗时: {t1-t0:.1f}s")

    # 帕累托前沿
    pf = [ind for ind in population if ind["rank"] == 0]
    print(f"\n  帕累托前沿规模: {len(pf)} 个解")

    # TOPSIS筛选
    result = topsis_select(pf)
    if result:
        best_ind, C_vals, best_idx = result
        print(f"\n  TOPSIS 推荐方案:")
        print(f"    H = {best_ind['x'][0]:.1f} m")
        print(f"    L = {best_ind['x'][1]:.1f} m")
        print(f"    x_start_buf = {best_ind['x'][2]:.1f} m")
        print(f"    F_buf = {best_ind['x'][3]:.1f} N")
        print(f"    目标值: f1(avg_vmax)={-best_ind['obj'][0]:.2f} m/s, "
              f"f2(var)={best_ind['obj'][1]:.4f}")

        cable_opt = ParabolicCable(W, best_ind["x"][0], R, L=best_ind["x"][1])
        res_verify = evaluate_solution(cable_opt, best_ind["x"][2], best_ind["x"][3], m_samples)
        print(f"\n  验证:")
        for m, vm, ve in zip(m_samples, res_verify["vmax_list"], res_verify["vend_list"]):
            print(f"    m={m:3.0f}kg: v_max={vm:.2f} m/s, v_end={ve:.2f} m/s")

        # 出图
        plot_fig6_pareto(population, best_ind)
        plot_fig7_three_weights(cable_opt, best_ind["x"][2], best_ind["x"][3])

    print("\n问题三全部图表已生成。")
