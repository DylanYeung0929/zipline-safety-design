#!/usr/bin/env python3
"""
蒙特卡洛鲁棒性分析
==================
Latin超立方抽样(LHS) × 1000次模拟,
评估推荐方案在参数不确定性下的安全通过率。
"""
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problem3_fixed import CatenaryFromH0, FastSimAdaptive, get_frontal_area, G
from problem2_dynamics import V_SAFE


def latin_hypercube_sample(n_samples, n_params):
    """生成Latin超立方样本 (0,1)区间"""
    samples = np.zeros((n_samples, n_params))
    for j in range(n_params):
        perm = np.random.permutation(n_samples)
        samples[:, j] = (perm + np.random.uniform(0, 1, n_samples)) / n_samples
    return samples


def mc_robustness_analysis(H=19.4, H0=64.9, d_buf=9.7,
                            n_samples=1000, seed=42):
    """
    蒙特卡洛鲁棒性分析。

    不确定参数:
      - 体重 m ~ U(28, 82)
      - 滚动摩擦 mu_roll ~ N(0.005, 0.001^2)
      - 空气阻力系数 C_d ~ N(0.7, 0.07^2)
      - 测速噪声 eps_v ~ N(0, 0.1^2)
      - 制动力执行误差 eps_B ~ N(0, 5^2)
    """
    np.random.seed(seed)

    W = 600.0; R = 1.2
    cable = CatenaryFromH0(W, H, R, H0 * 1e3)

    # LHS抽样
    lhs = latin_hypercube_sample(n_samples, 5)

    # 参数映射
    m_samples = 28.0 + lhs[:, 0] * 54.0          # U(28, 82)
    from scipy.stats import norm
    mu_roll_samples = norm.ppf(lhs[:, 1], loc=0.005, scale=0.001)
    cd_samples = norm.ppf(lhs[:, 2], loc=0.7, scale=0.07)
    eps_v_samples = norm.ppf(lhs[:, 3], loc=0.0, scale=0.1)
    eps_B_samples = norm.ppf(lhs[:, 4], loc=0.0, scale=5.0)

    # 分组统计: 与论文运营策略一致的体重区间
    #   30-40kg 辅助策略组, 40-80kg 主方案组
    groups = {'30-40kg': (30, 40), '40-80kg': (40, 80)}
    results = {g: {'safe': 0, 'stall': 0, 'over': 0, 'v_end': []} for g in groups}
    all_safe, all_stall, all_over = 0, 0, 0

    # 创建仿真器 (物理参数通过实例属性注入)
    sim = FastSimAdaptive(cable)

    # Monte Carlo仿真 (考虑参数不确定性和闭环噪声)
    for i in range(n_samples):
        m = m_samples[i]
        mu_roll = max(0.001, mu_roll_samples[i])  # 截断正值
        cd = max(0.1, cd_samples[i])
        eps_v = eps_v_samples[i]
        eps_B = eps_B_samples[i]

        # 注入本轮的物理参数
        sim.cd = cd
        sim.mu_roll = mu_roll

        try:
            xx, vv, _ = sim.simulate(m, x_buf=d_buf, eps_v=eps_v, eps_B=eps_B)

            if len(xx) > 0:
                v_end = vv[-1]
                reached = xx[-1] < 1.0 if sim._sgn < 0 else xx[-1] > (W - 1.0)

                if not reached or (len(vv) > 2 and np.min(vv[1:]) < 0.01):
                    status = 'stall'
                elif v_end > V_SAFE:
                    status = 'over'
                else:
                    status = 'safe'
            else:
                status = 'stall'
                v_end = 0.0
        except Exception:
            status = 'stall'
            v_end = 0.0

        # 分类
        if status == 'safe':
            all_safe += 1
        elif status == 'stall':
            all_stall += 1
        else:
            all_over += 1

        for gname, (mlo, mhi) in groups.items():
            if mlo <= m <= mhi:
                results[gname][status] += 1
                results[gname]['v_end'].append(v_end)
                break

    # 输出结果
    print(f"\n{'='*60}")
    print(f"蒙特卡洛鲁棒性分析结果 (N={n_samples})")
    print(f"{'='*60}")
    print(f"{'体重组':<10} {'P_safe':>8} {'P_stall':>8} {'P_over':>8} {'v_end_mean':>12} {'v_end_std':>10}")
    print(f"{'-'*60}")
    for gname, data in results.items():
        n_g = data['safe'] + data['stall'] + data['over']
        if n_g > 0:
            ps = data['safe'] / n_g * 100
            pst = data['stall'] / n_g * 100
            po = data['over'] / n_g * 100
            ve = np.array(data['v_end'])
            print(f"{gname:<10} {ps:7.1f}% {pst:7.1f}% {po:7.1f}% "
                  f"{ve.mean():11.3f} {ve.std():10.3f}")
    print(f"{'='*60}")
    print(f"全体重 (30-80): P_safe={all_safe/n_samples*100:.1f}%, "
          f"P_stall={all_stall/n_samples*100:.1f}%, "
          f"P_over={all_over/n_samples*100:.1f}%")

    return {
        'P_safe': all_safe / n_samples * 100,
        'P_stall': all_stall / n_samples * 100,
        'P_over': all_over / n_samples * 100,
        'group_results': results,
    }


if __name__ == "__main__":
    print("Latin超立方蒙特卡洛鲁棒性验证")
    print("参数: H=19.4m, H0=64.9kN, d_b=9.7m")
    result = mc_robustness_analysis(
        H=19.4, H0=64.9, d_buf=9.7,
        n_samples=1000, seed=42
    )
