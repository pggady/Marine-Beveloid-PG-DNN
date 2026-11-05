"""Fig. 11 redrawn from figures/Fig11_stats.json at the printed size (text width 6.5 in), so that the
font sizes in the manuscript are slightly smaller than the 10 pt body text."""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

stats_file, out = sys.argv[1], sys.argv[2]
D = json.load(open(stats_file, encoding='utf-8'))
base, st = D['baseline'], D['mape']
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
FS, FT = 8.5, 8          # labels; ticks, legends and annotations

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': FS,
                     'mathtext.fontset': 'stix', 'axes.unicode_minus': False, 'pdf.fonttype': 42})
fig, axes = plt.subplots(3, 3, figsize=(6.5, 6.1))
twins = []
for idx, ((name, label), ax) in enumerate(zip(PANELS, axes.flat)):
    sc = SCALE.get(name, 1.0)
    if name == 'kesai':
        texts = []
        outside = []
        for pair, tag, c1, c2 in (('pair1', 'Intersecting', 'tab:blue', 'tab:red'),
                                  ('pair2', 'Crossed', 'tab:purple', 'tab:orange')):
            r = {k: np.array(v) for k, v in st['kesai_' + pair]['points'].items()}
            ls = '-' if pair == 'pair1' else '--'
            for sgn in (-1, 1):
                m = np.sign(r['x']) == sgn
                ax.plot(r['x'][m], r['p'][m], ls, c=c1, lw=0.9, label=f'{tag}, solver' if sgn < 0 else None)
            ax.plot(r['x'], r['p_hat'], 'x', c=c2, ms=3, label=f'{tag}, PG-DNN')
            ax.axvline(base[pair]['kesai'], c=c1, ls=':', lw=0.8)
            outside += st['kesai_' + pair]['outside_x']
            texts.append(f"{tag[0]} {st['kesai_' + pair]['p']:.2f}%")
        xo = max(abs(v) for v in outside)
        ax.axvspan(-xo - 0.25, xo + 0.25, color='lightgray', alpha=0.6, lw=0)
        ax.set_ylabel('Max contact stress (MPa)')
        ax.legend(loc='upper center', fontsize=7, frameon=True, edgecolor='gray', ncol=1, handlelength=1.5,
                  borderpad=0.3, labelspacing=0.2)
        ax.text(1.0, 1.04, 'MAPE $p$ ' + ', '.join(texts), transform=ax.transAxes, fontsize=7.5, ha='right')
        ax.set_ylim(300, 1750)
    else:
        r = {k: np.array(v) for k, v in st[name]['points'].items()}
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
        ax2.tick_params(labelsize=FT)
        ax.text(1.0, 1.04, f"MAPE $p$ {st[name]['p']:.2f}%, $a$ {st[name]['a']:.2f}%",
                transform=ax.transAxes, fontsize=FT, ha='right')
        handles = l1 + l2 + l3 + l4
        twins.append((ax, ax2, r))
    ax.set_xlabel(label)
    ax.tick_params(labelsize=FT)
    ax.text(-0.02, 1.04, f'({chr(97 + idx)})', transform=ax.transAxes, fontsize=9, fontweight='bold', ha='right')
for i, keys in ((0, ('p', 'p_hat')), (1, ('a', 'a_hat'))):
    vals = np.concatenate([t[2][k] for t in twins for k in keys])
    lo, hi = vals.min(), vals.max()
    for t in twins:
        t[i].set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
fig.legend(handles, [h.get_label() for h in handles], loc='lower center', ncol=2, fontsize=FT,
           bbox_to_anchor=(0.5, -0.02), frameon=True, edgecolor='gray')
fig.subplots_adjust(hspace=0.7, wspace=0.62, bottom=0.13, left=0.1, right=0.9, top=0.96)
plt.savefig(out, bbox_inches='tight', pad_inches=0.02)
plt.savefig(out.rsplit('.', 1)[0] + '.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
print('saved', out)
