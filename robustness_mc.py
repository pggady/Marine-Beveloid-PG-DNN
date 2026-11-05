# 先导入PyTorch，确保在其他库之前加载（避免DLL冲突）
import torch  # noqa: F401
import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd

import gear_env_v2
from gear_env_v2 import GearEnvV2, predict_contact_stress_pinn, predict_contact_stress_solver
from tpe import VAR_NAMES
from system import create_example_system

"""
最优设计的鲁棒性蒙特卡洛分析（Reviewer 1 第 21 条）

三种扰动：
    vars      决策变量各自施加 ±rel 的均匀相对扰动（重新求解几何，PG-DNN 预测接触量）
    load_r    法向载荷施加 ±r 的均匀相对扰动（几何不变）
    combined  决策变量 ±rel 与载荷 ±10% 同时扰动
统计：p13、p23 相对名义值的最大增幅；名义限值（S_min ≥ 1.25、2a ≤ w·b）与
物理限值（S_min ≥ 1、2a ≤ b）的满足概率；接触应力增幅最大的若干样本用 Hertz 求解器复算。

用法：
    python robustness_mc.py --summary results_tpe/pinn_w0.8_seed42/summary.json
"""

PAIRS = {'pair1': 'p23', 'pair2': 'p13'}  # pair1 = 相交轴，pair2 = 交错轴


def contact(inputs, factor=1.0):
    """PG-DNN 预测某一齿轮对的接触量；factor 为载荷系数"""
    x = dict(inputs)
    x['Fn'] = inputs['Fn'] * factor
    r = predict_contact_stress_pinn(x)
    return r['Pmax'], r['a_len']


def evaluate_sample(env, params, factor):
    env.current_params.update(params)
    m = env._calculate_metrics()
    if not m.get('valid', False):
        return None
    out = {'min_safety': m['min_safety'], 'factor': factor, **params}
    for pair in PAIRS:
        out[f'Pmax_{pair}'], out[f'a_len_{pair}'] = contact(m[f'contact_inputs_{pair}'], factor)
        out[f'inputs_{pair}'] = m[f'contact_inputs_{pair}']
    return out


def statistics(df, nominal, widths, args):
    w_min = {'pair1': min(widths['b2'], widths['b3']), 'pair2': min(widths['b1'], widths['b3'])}
    res = {'n_valid': int(len(df))}
    for pair, name in PAIRS.items():
        inc = (df[f'Pmax_{pair}'] / nominal[f'Pmax_{pair}'] - 1) * 100
        res[f'max_increase_{name}_pct'] = float(inc.max())
        res[f'p95_increase_{name}_pct'] = float(np.percentile(inc, 95))
    nominal_ok = np.ones(len(df), bool)
    physical_ok = np.ones(len(df), bool)
    for pair in PAIRS:
        nominal_ok &= (2 * df[f'a_len_{pair}'] <= args.width_factor * w_min[pair]).values
        physical_ok &= (2 * df[f'a_len_{pair}'] <= w_min[pair]).values
    res['P_S_ge_1.25'] = float(np.mean(df['min_safety'] >= 1.25))
    res['P_S_ge_1'] = float(np.mean(df['min_safety'] >= 1.0))
    res['P_2a_le_wb'] = float(np.mean(nominal_ok))
    res['P_2a_le_b'] = float(np.mean(physical_ok))
    res['P_p_le_p0'] = float(np.mean(np.maximum(df['Pmax_pair1'], df['Pmax_pair2']) <= args.p0))
    return res


def main():
    parser = argparse.ArgumentParser(description='Monte Carlo robustness analysis of the optimized design')
    parser.add_argument('--summary', required=True, help='summary.json of a tpe.py run')
    parser.add_argument('--n', type=int, default=10000)
    parser.add_argument('--rel', type=float, default=0.01, help='relative perturbation of the decision variables')
    parser.add_argument('--loads', type=float, nargs='+', default=[0.1, 0.2])
    parser.add_argument('--combined-load', type=float, default=0.1)
    parser.add_argument('--worst', type=int, default=20, help='samples recomputed by the Hertzian solver')
    parser.add_argument('--width-factor', type=float, default=0.8)
    parser.add_argument('--p0', type=float, default=1347.76)  # sigma_HP by ISO 6336-2:2019 (KISSsoft 2026)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-dir', default='pgnn_model')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    args.out = args.out or os.path.join(os.path.dirname(args.summary), 'robustness')
    os.makedirs(args.out, exist_ok=True)
    gear_env_v2.PINN_MODEL_DIR = args.model_dir
    rng = np.random.default_rng(args.seed)

    best = json.load(open(args.summary, encoding='utf-8'))['best']
    if best is None:
        raise SystemExit('the run has no verified feasible design')
    params0 = {k: float(best['params'][k]) for k in VAR_NAMES}

    base_params = create_example_system()
    widths = {'b1': base_params['gear1']['b2'], 'b2': base_params['gear2']['b2'], 'b3': base_params['gear3']['b1']}
    env = GearEnvV2(base_params)
    nominal = evaluate_sample(env, params0, 1.0)

    def perturbed_params():
        return {k: v * (1 + rng.uniform(-args.rel, args.rel)) for k, v in params0.items()}

    cases = {f'vars_{args.rel:g}': lambda: (perturbed_params(), 1.0)}
    for r in args.loads:
        cases[f'load_{r:g}'] = lambda r=r: (None, 1 + rng.uniform(-r, r))
    cases[f'combined_{args.rel:g}_{args.combined_load:g}'] = \
        lambda: (perturbed_params(), 1 + rng.uniform(-args.combined_load, args.combined_load))

    summary = {'nominal': {k: v for k, v in nominal.items() if not k.startswith('inputs_')}, 'cases': {},
               'settings': vars(args)}
    for name, draw in cases.items():
        t0 = time.time()
        rows = []
        for _ in range(args.n):
            params, factor = draw()
            if params is None:  # 只扰动载荷：几何不变
                row = {k: v for k, v in nominal.items()}
                row['factor'] = factor
                for pair in PAIRS:
                    row[f'Pmax_{pair}'], row[f'a_len_{pair}'] = contact(nominal[f'inputs_{pair}'], factor)
            else:
                row = evaluate_sample(env, params, factor)
            if row is not None:
                rows.append(row)
        df = pd.DataFrame(rows)
        stats = statistics(df, nominal, widths, args)
        stats['n_failed_geometry'] = args.n - len(df)

        # 接触应力增幅最大的样本用 Hertz 求解器复算
        df['rel_increase'] = np.maximum(df['Pmax_pair1'] / nominal['Pmax_pair1'],
                                        df['Pmax_pair2'] / nominal['Pmax_pair2']) - 1
        worst = df.nlargest(args.worst, 'rel_increase')
        diffs = []
        for _, row in worst.iterrows():
            for pair in PAIRS:
                x = dict(row[f'inputs_{pair}'])
                x['Fn'] = x['Fn'] * row['factor']
                s = predict_contact_stress_solver(x)
                diffs.append(abs(row[f'Pmax_{pair}'] - s['Pmax']) / s['Pmax'] * 100)
        stats['worst_solver_max_diff_pct'] = float(np.nanmax(diffs)) if diffs else None
        stats['seconds'] = time.time() - t0
        summary['cases'][name] = stats
        df.drop(columns=[c for c in df.columns if c.startswith('inputs_')]).to_csv(
            os.path.join(args.out, f'mc_{name}.csv'), index=False)
        print(name, json.dumps(stats, ensure_ascii=False))

    with open(os.path.join(args.out, 'robustness_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
