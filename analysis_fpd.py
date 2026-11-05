"""
优化前后 FPD 角与接触应力变化的分解：FPD 角增大但接触应力减小的原因

对两个齿轮副，用 Hertz 求解器计算：
  - 优化前（I）、优化后（O）的曲率、综合曲率、A/B、接触椭圆和接触应力；
  - 只改变一组输入（FPD 角、螺旋角、节锥角）时的接触应力，按两种顺序累加，得到各组输入的贡献。

用法：
    python analysis_fpd.py --out logs/analysis_fpd.json
"""
import torch  # noqa: F401  （须在 MATLAB 引擎之前加载）
import argparse
import itertools
import json

import numpy as np

from contact_stress import calculate_hertz_contact_stress
from gear_env_v2 import GearEnvV2
from system import create_example_system

INITIAL = {'gama1': 5.0, 'gama2_gear1': 1.25, 'gama2_gear2': 2.0, 'beta3': 15.0, 'xt3': 0.0}
GROUPS = {'FPD angle': ('kesai',), 'helix angles': ('beta1', 'beta2'), 'pitch cone angles': ('gama1', 'gama2')}


def inputs(params):
    env = GearEnvV2(create_example_system())
    env.current_params.update(params)
    m = env._calculate_metrics()
    return {pair: dict(m['contact_inputs_' + pair]) for pair in ('pair1', 'pair2')}


def solve(x):
    """与 predict_contact_stress_solver 相同的输入转换，返回全部中间量"""
    r = calculate_hertz_contact_stress({
        'gama1': np.radians(x['gama1']), 'gama2': np.radians(x['gama2']),
        'beta1': np.radians(x['beta1']), 'beta2': np.radians(x['beta2']),
        'N1': int(x['N1']), 'N2': int(x['N2']), 'mn': x['mn'], 'an': np.radians(x['an']),
        'E': x['E'], 'nu': x['nu'], 'b1': 188.0, 'b2': 180.0, 'Fn': x['Fn'], 'kesai': x['kesai'],
        'sigma_rad': np.radians(x['kesai']), 'xt1': 0.0, 'xt2': 0.0, 'fuhao': -1, 'drive_gear': 1})
    k1, k2, eps = r['k1_1'], r['k2_1'], np.radians(x['kesai'])
    sumK = 0.5 * (k1 + k2)
    diffK = 0.5 * np.sqrt(k1 ** 2 + k2 ** 2 + 2 * k1 * k2 * np.cos(2 * eps))
    A, B = 0.5 * (sumK - diffK), 0.5 * (sumK + diffK)
    return {'p': r['Pmax'], 'a': r['a_len'], 'b': r['b_width'], 'area': np.pi * r['a_len'] * r['b_width'],
            'k_pinion': k1, 'k_wheel': k2, 'rho_pinion': 1 / k1, 'rho_wheel': 1 / k2,
            'sumK': 2 * sumK, 'A_over_B': A / B, 'b_over_a': r['b_width'] / r['a_len'], 'Fn': x['Fn']}


def mix(I, O, groups):
    x = dict(I)
    for g in groups:
        for k in GROUPS[g]:
            x[k] = O[k]
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', default='results_tpe/pinn_w0.8_seed42/summary.json')
    parser.add_argument('--out', default='logs/analysis_fpd.json')
    args = parser.parse_args()
    best = json.load(open(args.summary, encoding='utf-8'))['best']['params']
    XI, XO = inputs(INITIAL), inputs(best)
    out = {}
    for pair, name in (('pair1', 'intersecting (Gear 2 + Gear 3)'), ('pair2', 'crossed (Gear 1 + Gear 3)')):
        I, O = XI[pair], XO[pair]
        rI, rO = solve(I), solve(O)
        res = {'inputs_initial': I, 'inputs_optimized': O, 'initial': rI, 'optimized': rO,
               'only_one_group_changed': {}, 'shapley_contribution_pct': {}}
        for g in GROUPS:
            res['only_one_group_changed'][g] = solve(mix(I, O, [g]))
        # Shapley 分解：对所有改变顺序取平均的边际贡献（相对优化前 p 的百分比）
        names = list(GROUPS)
        contrib = {g: [] for g in names}
        for order in itertools.permutations(names):
            done = []
            p_prev = rI['p']
            for g in order:
                done.append(g)
                p_new = solve(mix(I, O, done))['p']
                contrib[g].append((p_new - p_prev) / rI['p'] * 100)
                p_prev = p_new
        res['shapley_contribution_pct'] = {g: float(np.mean(v)) for g, v in contrib.items()}
        res['total_change_pct'] = (rO['p'] - rI['p']) / rI['p'] * 100
        out[pair] = res
        print('=====', name)
        for k in ('kesai', 'beta1', 'beta2', 'gama1', 'gama2', 'Fn'):
            print(f"  {k:6s} {I[k]:10.4f} -> {O[k]:10.4f}")
        for k in ('rho_pinion', 'rho_wheel', 'sumK', 'A_over_B', 'b_over_a', 'a', 'b', 'area', 'p'):
            print(f"  {k:10s} {rI[k]:12.5g} -> {rO[k]:12.5g}  ({(rO[k] - rI[k]) / rI[k] * 100:+.2f}%)")
        for g, v in res['only_one_group_changed'].items():
            print(f"  only {g:18s}: p = {v['p']:8.2f} ({(v['p'] - rI['p']) / rI['p'] * 100:+.2f}%)")
        for g, v in res['shapley_contribution_pct'].items():
            print(f"  contribution of {g:18s}: {v:+.2f}%")
        print(f"  total change: {res['total_change_pct']:+.2f}%")
    json.dump(out, open(args.out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=float)


if __name__ == '__main__':
    main()
