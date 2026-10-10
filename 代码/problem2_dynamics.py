"""
问题二：非线性动力学与缓冲寻优
=================================
方法：四阶Runge--Kutta求解ODE + 网格搜索缓冲可行域
输出：相轨迹图、速度/加速度双轴图、缓冲可行域热力图
"""
import numpy as np
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
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

# ---- 路径 ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problem1_catenary import ElasticCatenary, G, RHO_FE, E_STEEL, FIG_DIR

# ============ 物理常数 ============
C_D        = 0.7        # 人体坐姿空气阻力系数 (参考: 自行车运动空气动力学)
RHO_AIR    = 1.225      # 空气密度 kg/m³ (海平面, 15°C)
A_FRONTAL  = 0.4        # 人体坐姿正面投影面积 m² (溜索坐姿)
MU_ROLL    = 0.005      # 滚动摩擦系数 (滑轮轴承+滚动阻力, 参考: SKF轴承手册)
MU_CABLE   = 0.025      # 钢缆-滑轮等效滑动摩擦系数 (高精度航空级滑轮组, 张力诱导)
L_CONTACT  = 0.06       # 滑轮与钢缆接触长度 m
V_SAFE     = 3.5        # 安全末速上限 m/s

def get_frontal_area(m):
    """异速生长迎风面积: A ∝ m^(2/3)
    参考: 生物力学异速生长法则 (Allometric scaling), A_ref=0.4m² 对应80kg成年人坐姿
    """
    return A_FRONTAL * (m / 80.0) ** (2.0 / 3.0)


class DynamicsSimulator:
    """溜索动力学仿真器 — RK4求解 + 缓冲优化"""

    def __init__(self, catenary: ElasticCatenary):
        self.ec = catenary
        self._build_interpolators()

    def _build_interpolators(self):
        """预计算钢缆形态插值函数 (用空载形态, 准静态近似, 加速RK4查询)"""
        W = self.ec.W
        self._x_grid = np.linspace(0, W, 2000)
        self._y      = self.ec.unloaded_curve(self._x_grid)
        self._yp     = self.ec.unloaded_slope(self._x_grid)
        self._T      = self.ec.unloaded_tension(self._x_grid)

        # 计算二阶导数 (曲率相关)
        self._ypp = np.gradient(self._yp, self._x_grid)
        # 平滑边界
        self._ypp[0] = self._ypp[1]
        self._ypp[-1] = self._ypp[-2]

        # 局部切线角
        self._theta = np.arctan(self._yp)

        # 曲率 κ = |y''| / (1 + y'²)^(3/2)
        self._kappa = np.abs(self._ypp) / (1 + self._yp**2)**1.5

        # 插值器
        self._f_y    = interp1d(self._x_grid, self._y,    kind='cubic', bounds_error=False, fill_value='extrapolate')
        self._f_yp   = interp1d(self._x_grid, self._yp,   kind='cubic', bounds_error=False, fill_value='extrapolate')
        self._f_ypp  = interp1d(self._x_grid, self._ypp,  kind='cubic', bounds_error=False, fill_value='extrapolate')
        self._f_T    = interp1d(self._x_grid, self._T,    kind='cubic', bounds_error=False, fill_value='extrapolate')
        self._f_theta = interp1d(self._x_grid, self._theta, kind='cubic', bounds_error=False, fill_value='extrapolate')
        self._f_kappa = interp1d(self._x_grid, self._kappa, kind='cubic', bounds_error=False, fill_value='extrapolate')

    def friction_force(self, x: float, m: float):
        """摩擦力: F_fric = μ_roll·mg·cosθ + μ_cable·T·κ·L_contact
        第一项: 滚动阻力, 正比于重力法向分量 (SKF轴承手册)
        第二项: 钢缆张力通过曲率诱发的附加法向压力, 类似皮带轮效应
        """
        theta = float(self._f_theta(x))
        T_val = float(self._f_T(x))
        kappa_val = float(self._f_kappa(x))
        N_gravity = m * G * np.cos(theta)
        N_tension = T_val * kappa_val * L_CONTACT
        return MU_ROLL * N_gravity + MU_CABLE * N_tension

    def air_drag(self, v: float, m: float):
        """非线性空气阻力 F = 0.5*Cd*ρ*A(m)*v² (含方向, 迎风面积依异速生长法则)"""
        A = get_frontal_area(m)
        return 0.5 * C_D * RHO_AIR * A * v * abs(v)

    def total_force(self, x: float, v: float, m: float,
                    x_buf: float = None, F_buf: float = 0.0):
        """计算总切向力 (带方向的代数和)

        F_gravity: -mg*sinθ (势能梯度, 已带符号)
        F_fric:    μ*N, 幅值, 始终与v反向 → -sign(v)*F_fric
        F_air:     0.5*Cd*ρ*A*v*|v|, 已带符号(与v反向)
        F_buffer:  F_buf*sign(v), 已带符号(与v反向)
        """
        theta = float(self._f_theta(x))
        F_gravity = -m * G * np.sin(theta)  # 势能梯度, 已带符号
        F_fric_mag = self.friction_force(x, m)  # 幅值, 正比于 mg·cosθ
        F_air = self.air_drag(v, m)        # 已带符号 (v*|v|), 迎风面积依体重缩放

        F_buffer = 0.0
        if x_buf is not None and x >= x_buf:
            F_buffer = F_buf * np.sign(v) if v != 0 else 0.0

        # F_air 和 F_buffer 已带符号(与v反向), F_fric需要手动加符号
        sgn = np.sign(v) if abs(v) > 1e-9 else np.sign(F_gravity)
        F_total = F_gravity - sgn * F_fric_mag - F_air - F_buffer
        return np.clip(F_total, -1e6, 1e6)

    def ode_system(self, t: float, u: list, m: float,
                   x_buf: float = None, F_buf: float = 0.0):
        """ODE: du/dt = f(t, u),  u = [x, v]
        dx/dt = v·cos(θ)  — 水平位移与切向速度的几何关系
        dv/dt = a_tan     — 切向加速度
        """
        x, v = u
        theta = float(self._f_theta(x))
        a = self.total_force(x, v, m, x_buf, F_buf) / m
        return [v * np.cos(theta), a]

    def rk4_step(self, t: float, u: np.ndarray, dt: float, m: float,
                 x_buf: float = None, F_buf: float = 0.0):
        """单步RK4推进"""
        f = lambda t_, u_: self.ode_system(t_, u_, m, x_buf, F_buf)

        k1 = np.array(f(t, u))
        k2 = np.array(f(t + dt/2, u + dt/2 * k1))
        k3 = np.array(f(t + dt/2, u + dt/2 * k2))
        k4 = np.array(f(t + dt,   u + dt   * k3))

        return u + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    def simulate(self, m: float, dt: float = 0.05,
                 x_buf: float = None, F_buf: float = 0.0,
                 max_t: float = 120.0, x0: float = None, v0: float = 0.0):
        """完整仿真: 自动从高端点出发, RK4自适应积分

        若 x0=None, 自动选择高端点作为起点 (H>0→x=W, H<0→x=0)
        """
        W = self.ec.W
        if x0 is None:
            x0 = W if self.ec.H > 0 else 0.0

        # 确定方向: 从高到低
        dir_sign = -1.0 if x0 > W/2 else 1.0   # x0=W → 向左; x0=0 → 向右
        x_end = 0.0 if x0 > W/2 else W

        t = 0.0
        u = np.array([x0, v0], dtype=float)
        trajectory = [(t, u[0], u[1])]

        while t < max_t:
            adapt_dt = min(dt, 0.5 / (1.0 + 0.1 * abs(u[1])))
            u_new = self.rk4_step(t, u, adapt_dt, m, x_buf, F_buf)

            # 边界检测
            if dir_sign < 0:  # 向左滑行
                if u_new[0] <= x_end:
                    u_new[0] = x_end
                    u = u_new
                    t += adapt_dt
                    trajectory.append((t, u[0], u[1]))
                    break
            else:  # 向右滑行
                if u_new[0] >= x_end:
                    u_new[0] = x_end
                    u = u_new
                    t += adapt_dt
                    trajectory.append((t, u[0], u[1]))
                    break

            # 速度爆炸检测
            if abs(u_new[1]) > 150.0:
                break

            u = u_new
            t += adapt_dt
            trajectory.append((t, u[0], u[1]))

            # 停滞检测 (给初始加速时间)
            if abs(u[1]) < 0.003 and t > 2.0 and u[0] > x_end + 2 and u[0] < x0 - 2:
                break

        arr = np.array(trajectory)
        self._traj_dir = dir_sign
        return arr  # (N, 3): t, x, v

    def get_vmax_and_vend(self, traj: np.ndarray):
        """从轨迹提取: 最大速度, 末速, 最大速度位置"""
        v = traj[:, 2]
        x = traj[:, 1]
        v_abs = np.abs(v)
        idx_max = np.argmax(v_abs)
        W = self.ec.W
        x_end = 0.0 if getattr(self, '_traj_dir', -1) < 0 else W
        reached = abs(x[-1] - x_end) < 0.5
        return {
            "v_max":  v_abs[idx_max],
            "v_end":  abs(v[-1]),
            "x_vmax": x[idx_max],
            "t_total": traj[-1, 0] if reached else None
        }


# ============ 缓冲可行域搜索 ============
def buffer_feasibility_search(sim: DynamicsSimulator, m_list: list,
                               x_range: tuple = (200, 580),
                               f_range: tuple = (10, 200),
                               n_x: int = 40, n_f: int = 40):
    """网格搜索缓冲可行域"""
    x_vals = np.linspace(x_range[0], x_range[1], n_x)
    f_vals = np.linspace(f_range[0], f_range[1], n_f)
    score_grid = np.zeros((n_f, n_x))

    print(f"  缓冲搜索: {len(m_list)}个体重 × {n_x}×{n_f}网格 = {len(m_list)*n_x*n_f}次仿真...")

    for i, Fb in enumerate(f_vals):
        for j, xs in enumerate(x_vals):
            safe_count = 0
            total_vmax = 0.0
            for m in m_list:
                traj = sim.simulate(m, x_buf=xs, F_buf=Fb)
                res  = sim.get_vmax_and_vend(traj)
                vend = res["v_end"]
                vmax = res["v_max"]
                reached = res["t_total"] is not None
                if vend <= V_SAFE and vmax > 0.5 and reached:
                    safe_count += 1
                    total_vmax += vmax

            safe_rate = safe_count / len(m_list)
            avg_vmax  = total_vmax / max(safe_count, 1)
            score_grid[i, j] = safe_rate * avg_vmax if safe_count == len(m_list) else safe_rate * 0.5

    return x_vals, f_vals, score_grid


# ============ 图3: 速度-位移相轨迹 (三曲线对比) ============
def plot_fig3_phase(sim: DynamicsSimulator, m_test=75.0,
                    x_buf_opt=None, F_buf_opt=None,
                    x_buf_early=None, F_buf_early=None):
    """图3: 无缓冲/过早缓冲/最优缓冲 三曲线对比 (含危险区)"""
    traj_nb = sim.simulate(m_test)
    traj_opt = sim.simulate(m_test, x_buf=x_buf_opt, F_buf=F_buf_opt) if x_buf_opt else None
    traj_early = sim.simulate(m_test, x_buf=x_buf_early, F_buf=F_buf_early) if x_buf_early else None

    fig, ax = plt.subplots(figsize=(12, 7))

    # 危险区填充 (速度取绝对值, 左向为负)
    v_nb_abs = np.abs(traj_nb[:, 2])
    v_max_all = np.max(v_nb_abs)
    if traj_opt is not None:
        v_max_all = max(v_max_all, np.max(np.abs(traj_opt[:, 2])))
    if traj_early is not None:
        v_max_all = max(v_max_all, np.max(np.abs(traj_early[:, 2])))
    ax.fill_between([0, sim.ec.W], V_SAFE, v_max_all * 1.1, alpha=0.10, color='#D62828')
    ax.axhline(y=V_SAFE, color='#D62828', linestyle='--', linewidth=1.5, alpha=0.7)

    # 三条曲线 (使用绝对速度)
    x_nb, v_nb = traj_nb[:, 1], np.abs(traj_nb[:, 2])
    ax.plot(x_nb, v_nb, '-', color='#2F6690', linewidth=2.2, label='无缓冲 (危险)')

    if traj_early is not None:
        x_e, v_e = traj_early[:, 1], np.abs(traj_early[:, 2])
        ax.plot(x_e, v_e, '--', color='#F4A261', linewidth=1.8, label='过早缓冲 (效率低)')

    if traj_opt is not None:
        x_o, v_o = traj_opt[:, 1], np.abs(traj_opt[:, 2])
        ax.plot(x_o, v_o, '-', color='#2A9D8F', linewidth=2.5, label='最优缓冲 (推荐)')
        # 标记最优缓冲启动位置
        ax.axvline(x=x_buf_opt, color='#2A9D8F', linestyle=':', linewidth=1.2, alpha=0.5,
                   label=f'缓冲启动 (x={x_buf_opt:.0f}m)')
        # 终点星标
        ax.plot(x_o[-1], v_o[-1], '*', color='#F2C14E', markersize=18, markeredgecolor='#333',
                markeredgewidth=1.0, zorder=10, label=f'末速 {v_o[-1]:.1f} m/s')

    # 标注危险区文字
    ax.text(sim.ec.W * 0.5, V_SAFE + 0.8, '危险区 ($v > 3.5$ m/s)', fontsize=11,
            color='#D62828', alpha=0.7, ha='center', style='italic')

    ax.set_xlabel("水平位移 $x$ (m)", fontsize=13)
    ax.set_ylabel("滑行速度 $v$ (m/s)", fontsize=13)
    ax.set_title("不同缓冲策略下的速度—位移相轨迹对比", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.set_xlim(0, sim.ec.W)
    ax.set_ylim(0, v_max_all * 1.15)
    ax.grid(True, alpha=0.22)

    path = os.path.join(FIG_DIR, "fig03_phase_trajectory.pdf")
    fig.savefig(path, dpi=320, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(path.replace('.pdf', '.png'), dpi=320, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"[图3] 已保存: {path}")
    return path


# ============ 图4: 时间-速度/加速度双轴图 ============
def plot_fig4_double_axis(sim: DynamicsSimulator, m_test=75.0):
    """图4: 速度/加速度双轴时间图"""
    traj = sim.simulate(m_test)
    t, x, v = traj[:, 0], traj[:, 1], traj[:, 2]

    # 计算加速度 (用速度绝对值)
    v_abs = np.abs(v)
    a = np.gradient(v_abs, t)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 主轴: 速度
    color_v = '#2196F3'
    ax1.plot(t, v_abs, '-', color=color_v, linewidth=1.8, label="速度 $v(t)$")
    ax1.set_xlabel("时间 t (s)", fontsize=13)
    ax1.set_ylabel("速度 v (m/s)", color=color_v, fontsize=13)
    ax1.tick_params(axis='y', labelcolor=color_v)
    ax1.axhline(y=V_SAFE, color='#D62828', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_ylim(0, max(v_abs)*1.2)

    # 副轴: 加速度
    ax2 = ax1.twinx()
    color_a = '#FF5722'
    ax2.plot(t, a, '-', color=color_a, linewidth=1.2, alpha=0.8, label="加速度 $a(t)$")
    ax2.set_ylabel("加速度 a (m/s²)", color=color_a, fontsize=13)
    ax2.tick_params(axis='y', labelcolor=color_a)
    ax2.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)

    # 图例合并
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    ax1.set_title("时间—速度 / 加速度双轴变化图", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(0, t[-1])

    path = os.path.join(FIG_DIR, "fig04_double_axis.pdf")
    fig.savefig(path, dpi=320, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(path.replace('.pdf', '.png'), dpi=320, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"[图4] 已保存: {path}")
    return path


# ============ 图5: 缓冲可行域热力图 ============
def plot_fig5_buffer_heatmap(x_vals, f_vals, score_grid, best_x=None, best_f=None):
    """图5: 缓冲启动可行域热力图"""
    X, Y = np.meshgrid(x_vals, f_vals)

    fig, ax = plt.subplots(figsize=(10, 7))

    # 只有全员通过才显示彩色
    Z_masked = np.where(score_grid > 0, score_grid, np.nan)
    im = ax.pcolormesh(X, Y, Z_masked, cmap='YlGnBu', shading='auto')

    # 不可行域 (灰色)
    infeasible = np.where(np.isnan(Z_masked), 1, np.nan)
    ax.pcolormesh(X, Y, infeasible, cmap='Greys', shading='auto', alpha=0.15,
                  vmin=0, vmax=1)

    cbar = fig.colorbar(im, ax=ax, label="综合评分", shrink=0.85)
    ax.set_xlabel("缓冲启动位置 $x_{\\rm start}^{\\rm buf}$ (m)", fontsize=13)
    ax.set_ylabel("缓冲摩擦力 $F_{\\rm buf}$ (N)", fontsize=13)
    ax.set_title("缓冲启动可行域热力图", fontsize=14, fontweight='bold')

    # 标记最优点
    if best_x is not None and best_f is not None:
        ax.plot(best_x, best_f, 'b*', markersize=20, markeredgecolor='white',
                markeredgewidth=1.5, zorder=5)
        ax.annotate(f"最优 ({best_x:.0f}m, {best_f:.0f}N)",
                    xy=(best_x, best_f), xytext=(best_x+20, best_f+15),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                    fontsize=11, fontweight='bold', color='blue')

    path = os.path.join(FIG_DIR, "fig05_buffer_heatmap.pdf")
    fig.savefig(path, dpi=320, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(path.replace('.pdf', '.png'), dpi=320, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"[图5] 已保存: {path}")
    return path


# ============ 主程序 ============
if __name__ == "__main__":
    # ---- 参数 ----
    # 问题二演示: 使用CatenaryFromH0直接指定水平张力, 获得紧缆形态 (与问题三一致)
    from problem3_fixed import CatenaryFromH0 as CH0

    W, H_val, R = 600.0, 20.0, 1.2
    H0_target_kN = 63.0  # kN

    # 用CatenaryFromH0创建紧缆 (sag≈8m), 适配DynamicsSimulator接口
    ch0 = CH0(W, H_val, R, H0_target_kN * 1e3)

    class CableAdapter:
        """将CatenaryFromH0适配为DynamicsSimulator所需接口"""
        def __init__(self, ch0):
            self.W = ch0.W; self.H = ch0.H; self.R = ch0.R
            self.H0 = ch0.H0; self.a = ch0.a
            self.x0 = ch0.x0
        def unloaded_curve(self, x): return ch0.y(np.asarray(x))
        def unloaded_slope(self, x): return ch0.yp(np.asarray(x))
        def unloaded_tension(self, x): return ch0.tension(np.asarray(x))

    ec = CableAdapter(ch0)
    m_person, x_person = 75.0, 300.0

    print("=" * 60)
    print("问题二：非线性动力学仿真与缓冲寻优")
    print("=" * 60)

    # 1. 先求解弹性悬链线
    print("\n[Step 1] 悬链线形态 (CatenaryFromH0)...")
    xf = np.linspace(0, ec.W, 500); yf = ec.unloaded_curve(xf)
    sag = np.max(ec.H * xf / ec.W - yf)
    print(f"  H0 = {ec.H0/1e3:.2f} kN, sag_max ≈ {sag:.1f} m")

    # 2. 建立动力学仿真器
    print("\n[Step 2] 动力学仿真...")
    sim = DynamicsSimulator(ec)

    # 无缓冲仿真
    traj_nb = sim.simulate(75.0)
    res_nb  = sim.get_vmax_and_vend(traj_nb)
    t_str = f"t={res_nb['t_total']:.1f}s" if res_nb['t_total'] is not None else "未到达终点"
    print(f"  无缓冲 (75kg): v_max={res_nb['v_max']:.2f} m/s @ x={res_nb['x_vmax']:.0f}m, "
          f"v_end={res_nb['v_end']:.2f} m/s, {t_str}")

    # 不同体重
    for m in [30, 55, 80]:
        traj = sim.simulate(m)
        res  = sim.get_vmax_and_vend(traj)
        print(f"  无缓冲 ({m}kg): v_max={res['v_max']:.2f} m/s, v_end={res['v_end']:.2f} m/s")

    # 3. 缓冲可行域搜索
    print("\n[Step 3] 缓冲可行域搜索...")
    m_samples = np.linspace(30, 80, 5)  # 5个采样体重 (加速)
    x_buf_range = (540, 595)   # 距终点 5~60m
    f_buf_range = (15, 80)     # 恒定缓冲摩擦力范围 (N)
    x_vals, f_vals, score_grid = buffer_feasibility_search(
        sim, list(m_samples), x_range=x_buf_range, f_range=f_buf_range,
        n_x=15, n_f=15)

    # 找最优点
    best_idx = np.unravel_index(np.nanargmax(score_grid), score_grid.shape)
    best_f, best_x = f_vals[best_idx[0]], x_vals[best_idx[1]]
    print(f"  最优缓冲参数: x_start = {best_x:.1f} m, F_buf = {best_f:.1f} N")

    # 验证最优参数
    print(f"\n[Step 4] 最优参数验证:")
    for m in [30, 45, 55, 65, 75, 80]:
        traj = sim.simulate(m, x_buf=best_x, F_buf=best_f)
        res  = sim.get_vmax_and_vend(traj)
        status = "OK" if res["v_end"] <= V_SAFE and res["t_total"] is not None else "FAIL"
        t_str = f"t={res['t_total']:.1f}s" if res['t_total'] else "未到终点"
        print(f"  m={m:3.0f}kg: v_max={res['v_max']:.2f} m/s, v_end={res['v_end']:.2f} m/s, {t_str} [{status}]")

    # 4. 出图
    print("\n[Step 5] 生成图表...")
    plot_fig3_phase(sim, m_test=75.0,
                    x_buf_opt=best_x, F_buf_opt=best_f,
                    x_buf_early=min(500, best_x * 2.2), F_buf_early=best_f * 1.5)
    plot_fig4_double_axis(sim, m_test=75.0)
    plot_fig5_buffer_heatmap(x_vals, f_vals, score_grid, best_x, best_f)

    print("\n问题二全部图表已生成。")
