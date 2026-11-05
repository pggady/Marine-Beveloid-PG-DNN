"""
Fig. 11：PG-DNN 与 Hertz 求解器的参数化对比（围绕优化设计逐个改变输入）
    (a)–(e)、(g)–(i)：相交副（齿轮 3 + 齿轮 2）的 p 和 a
    (f)：FPD 角（带符号）对两个齿轮副 p 的影响，验证 p 随 |ε| 增大

用法：
    python plot_fig11.py --summary results_tpe/pinn_w0.8_seed42/summary.json --out figures/Fig11.pdf
"""
import torch  # noqa: F401  （须在 MATLAB 引擎之前加载）
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

import gear_env_v2
from gear_env_v2 import GearEnvV2, predict_contact_stress_pinn, predict_contact_stress_solver
from system import create_example_system

# 与 LHS 数据范围一致
RANGES = {'gama1': (0.1, 5), 'gama2': (2, 8), 'beta1': (-36, 36), 'beta2': (-36, 36), 'N1': (17, 39),
          'N2': (40, 89), 'mn': (2, 8), 'Fn': (1e4, 1e5), 'kesai': (-12, 12), 'an': (18, 25),
          'E': (1.8e5, 2.2e5), 'nu': (0.2, 0.4)}
PANELS = [('gama1', r'Pitch cone angle of pinion $\gamma_i$ (deg)'),
          ('beta1', r'Helix angle of pinion $\beta_i$ (deg)'),
          ('N1', r'Number of teeth of pinion $N_i$'),
          ('mn', r'Normal module $m_n$ (mm)'),
          ('Fn', r'Normal load $F_n$ (kN)'),
          ('kesai', r'FPD angle $\varepsilon_{ij}^{L(R)}$ (deg)'),
          ('an', r'Normal pressure angle $\alpha_n$ (deg)'),
          ('E', r"Young's modulus $E$ (GPa)"),
          ('nu', r"Poisson's ratio $\nu$")]
SCALE = {'Fn': 1e-3, 'E': 1e-3}


def baseline_inputs(summary):
    """优化设计两个齿轮副的 PG-DNN 输入"""
    params = json.load(open(summary, encoding='utf-8'))['best']['params']
    env = GearEnvV2(create_example_system())
    env.current_params.update(params)
    m = env._calculate_metrics()
    return {k: {n: float(v) for n, v in m['contact_inputs_' + k].items()} for k in ('pair1', 'pair2')}


def sweep(base, name, values):
    """逐点计算求解器和 PG-DNN 的 p、a（求解器无效的点跳过）"""
    out = {k: [] for k in ('x', 'p', 'a', 'p_hat', 'a_hat')}
    for v in values:
        x = dict(base)
        x[name] = float(v)
        s = predict_contact_stress_solver(x)
        if not s['valid'] or not np.isfinite(s['Pmax']):
            continue
        h = predict_contact_stress_pinn(x)
        for k, val in zip(out, (v, s['Pmax'], s['a_len'], h['Pmax'], h['a_len'])):
            out[k].append(val)
    return {k: np.array(v) for k, v in out.items()}


def values_for(name, base_val, n=21):
    lo, hi = RANGES[name]
    if name == 'kesai':
        return np.linspace(lo, hi, 49)
    a, b = sorted((base_val * 0.75, base_val * 1.25))
    a, b = max(a, lo), min(b, hi)
    if name in ('N1', 'N2'):
        return np.unique(np.round(np.linspace(a, b, n)).astype(int))
    return np.linspace(a, b, n)


def mape(y_hat, y):
    return float(np.mean(np.abs(y_hat - y) / np.abs(y)) * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', default='results_tpe/pinn_w0.8_seed42/summary.json')
    parser.add_argument('--out', default='figures/Fig11.pdf')
    parser.add_argument('--model-dir', default='pgnn_model')
    args = parser.parse_args()
    gear_env_v2.PINN_MODEL_DIR = args.model_dir

    base = baseline_inputs(args.summary)
    print('baseline:', json.dumps(base, indent=1))
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 8,
                         'mathtext.fontset': 'stix', 'axes.unicode_minus': False})
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.0))
    stats = {}
    twins = []  # (p 轴, a 轴, 数据)：除 (f) 外各子图使用相同的纵轴范围
    for idx, ((name, label), ax) in enumerate(zip(PANELS, axes.flat)):
        sc = SCALE.get(name, 1.0)
        if name == 'kesai':
            texts = []
            for pair, tag, c1, c2 in (('pair1', 'Intersecting', 'tab:blue', 'tab:red'),
                                      ('pair2', 'Crossed', 'tab:purple', 'tab:orange')):
                r = sweep(base[pair], name, values_for(name, base[pair][name]))
                # ε = 0 附近为线接触，求解器无解：左右两支分开画
                ls = '-' if pair == 'pair1' else '--'
                for sgn in (-1, 1):
                    m = np.sign(r['x']) == sgn
                    ax.plot(r['x'][m], r['p'][m], ls, c=c1, lw=0.9, label=f'{tag}, solver' if sgn < 0 else None)
                ax.plot(r['x'], r['p_hat'], 'x', c=c2, ms=3, label=f'{tag}, PG-DNN')
                ax.axvline(base[pair][name], c=c1, ls=':', lw=0.8)
                # 训练数据剔除了 a > 100 mm 或 p < 400 MPa 的样本，MAPE 只在训练输出范围内统计
                inside = (r['a'] <= 100) & (r['p'] >= 400) & (r['p'] <= 2500)
                stats[f'kesai_{pair}'] = {'p': mape(r['p_hat'][inside], r['p'][inside]),
                                          'a': mape(r['a_hat'][inside], r['a'][inside]),
                                          'n': int(inside.sum()), 'points': {k: v.tolist() for k, v in r.items()},
                                          'outside_x': r['x'][~inside].tolist()}
                texts.append(f"{tag[0]}: {stats[f'kesai_' + pair]['p']:.2f}%")
            xo = max(abs(v) for k in ('pair1', 'pair2') for v in stats['kesai_' + k]['outside_x'])
            ax.axvspan(-xo - 0.25, xo + 0.25, color='lightgray', alpha=0.6, lw=0)
            ax.set_ylabel('Max contact stress (MPa)')
            ax.legend(loc='upper center', fontsize=5.5, frameon=True, edgecolor='gray', ncol=1)
            ax.text(1.0, 1.04, 'MAPE $p$ ' + ', '.join(texts), transform=ax.transAxes, fontsize=6.5, ha='right')
        else:
            r = sweep(base['pair1'], name, values_for(name, base['pair1'][name]))
            x = r['x'] * sc
            l1 = ax.plot(x, r['p'], '-', c='tab:blue', lw=0.9, label='Max contact stress, solver')
            l2 = ax.plot(x, r['p_hat'], 'x', c='tab:red', ms=3, label='Max contact stress, PG-DNN')
            ax2 = ax.twinx()
            l3 = ax2.plot(x, r['a'], '-', c='tab:green', lw=0.9, label='Semi-major axis, solver')
            l4 = ax2.plot(x, r['a_hat'], 's', c='tab:orange', ms=2, mfc='none', label='Semi-major axis, PG-DNN')
            ax.axvline(base['pair1'][name] * sc, c='gray', ls=':', lw=0.8)
            if idx % 3 == 0:
                ax.set_ylabel('Max contact stress (MPa)')
            if idx % 3 == 2:
                ax2.set_ylabel('Semi-major axis (mm)')
            ax2.tick_params(labelsize=7)
            stats[name] = {'p': mape(r['p_hat'], r['p']), 'a': mape(r['a_hat'], r['a']), 'n': int(len(r['x'])),
                           'points': {k: v.tolist() for k, v in r.items()}}
            ax.text(1.0, 1.04, f"MAPE $p$ {stats[name]['p']:.2f}%, $a$ {stats[name]['a']:.2f}%",
                    transform=ax.transAxes, fontsize=6.5, ha='right')
            handles = l1 + l2 + l3 + l4
            twins.append((ax, ax2, r))
        ax.set_xlabel(label)
        ax.tick_params(labelsize=7)
        ax.text(-0.02, 1.04, f'({chr(97 + idx)})', transform=ax.transAxes, fontsize=9, fontweight='bold')
    for i, keys in ((0, ('p', 'p_hat')), (1, ('a', 'a_hat'))):
        vals = np.concatenate([t[2][k] for t in twins for k in keys])
        lo, hi = vals.min(), vals.max()
        for t in twins:
            t[i].set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
    fig.legend(handles, [h.get_label() for h in handles], loc='lower center', ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='gray')
    fig.subplots_adjust(hspace=0.5, wspace=0.75, bottom=0.1)
    plt.savefig(args.out, bbox_inches='tight')
    plt.savefig(args.out.rsplit('.', 1)[0] + '.png', dpi=300, bbox_inches='tight')
    json.dump({'baseline': base, 'mape': stats}, open(args.out.rsplit('.', 1)[0] + '_stats.json', 'w'), indent=1)
    print(json.dumps({k: {a: b for a, b in v.items() if a != 'points'} for k, v in stats.items()}, indent=1))


if __name__ == '__main__':
    main()
