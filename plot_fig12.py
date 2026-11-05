"""
Fig. 12：TPE 优化过程（种子 42）
    (a) 目标函数收敛曲线（可行试验 + 历史最优）
    (b) 最小安全系数与目标函数的分布（含不可行试验）
    (c) 每 100 次试验中通过几何预检、PG-DNN 筛选和求解器验证的比例

用法：
    python plot_fig12.py --csv results_tpe/pinn_w0.8_seed42/optuna_results.csv --out Fig12.pdf
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='results_tpe/pinn_w0.8_seed42/optuna_results.csv')
    parser.add_argument('--out', default='Fig12.pdf')
    args = parser.parse_args()

    df = pd.read_csv(args.csv).sort_values('number')
    feas = df[df.stage == 'solver_feasible']
    infeas = df[df.stage.isin(['surrogate_infeasible', 'solver_infeasible']) & df.value.notna()]
    best = df.value.where(df.stage == 'solver_feasible').ffill().cummin()

    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'],
                         'axes.unicode_minus': False, 'font.size': 8.5, 'mathtext.fontset': 'stix', 'pdf.fonttype': 42,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5, 'axes.linewidth': 0.7})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.6, 2.4))

    # (a) 收敛曲线
    ax1.scatter(feas.number, feas.value, c='tab:blue', alpha=0.5, s=4, lw=0, label='Feasible trials')
    ax1.plot(df.number, best, c='red', lw=1.2, label='Best feasible')
    ax1.set_xlabel('Trial number')
    ax1.set_ylabel('Objective $O_b$ (MPa)')
    ax1.set_ylim(1600, 1800)
    ax1.legend(loc='upper right', frameon=True, edgecolor='black', borderpad=0.3, labelspacing=0.25)

    # (b) 安全系数与目标函数（不可行试验为 PG-DNN 预测值）
    ax2.scatter(infeas.min_safety, infeas.value, c='lightgray', s=4, lw=0, label='Infeasible trials')
    sc = ax2.scatter(feas.min_safety, feas.value, c=feas.number, cmap='viridis', alpha=0.7, s=4, lw=0,
                     label='Feasible trials')
    b = feas.loc[feas.value.idxmin()]
    ax2.scatter([b.min_safety], [b.value], color='red', marker='*', s=60, zorder=5, label='Best design')
    ax2.axvline(1.25, color='red', ls='--', lw=1, label='$S_{\\min}=1.25$')
    ax2.set_xlim(0, 2.5)
    ax2.set_ylim(1600, 2000)
    ax2.set_xlabel('Minimum safety factor $S_{\\min}$')
    ax2.set_ylabel('Objective $O_b$ (MPa)')
    ax2.legend(loc='upper right', frameon=True, edgecolor='black', borderpad=0.3, labelspacing=0.25, handlelength=1.5)
    fig.colorbar(sc, ax=ax2).set_label('Trial number')

    # (c) 三级评价通过比例
    blk = df.number.sub(1) // 100
    g = df.groupby(blk)
    frac = pd.DataFrame({
        'Geometric pre-check': g.stage.apply(lambda s: (s != 'pruned_geometry').mean()),
        'PG-DNN screening': g.stage.apply(lambda s: s.isin(['solver_feasible', 'solver_infeasible']).mean()),
        'Hertzian solver': g.stage.apply(lambda s: (s == 'solver_feasible').mean()),
    }) * 100
    x = (frac.index + 1) * 100
    for (name, col), m in zip(frac.items(), ['o', 's', '^']):
        ax3.plot(x, col, marker=m, ms=3.5, lw=1, label=name)
    ax3.set_xticks(x[1::2])
    ax3.set_xlabel('Trials (block of 100)')
    ax3.set_ylabel('Fraction passed (%)')
    ax3.set_ylim(0, 105)
    ax3.legend(loc='lower right', frameon=True, edgecolor='black', borderpad=0.3, labelspacing=0.25, handlelength=1.5)

    for ax, tag in zip((ax1, ax2, ax3), 'abc'):
        ax.text(-0.2, 1.03, f'({tag})', transform=ax.transAxes, fontsize=9, fontweight='bold', ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=0.3, w_pad=0.8)
    plt.savefig(args.out, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(args.out.rsplit('.', 1)[0] + '.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
    print('saved', args.out)


if __name__ == '__main__':
    main()
