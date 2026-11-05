# 先导入PyTorch，确保在其他库之前加载（避免DLL冲突）
import torch  # noqa: F401
import argparse
import csv
import json
import os
import time

import numpy as np

import gear_env_v2
from tpe import BOUNDS, STATS, VAR_NAMES, evaluate_design, get_bounds
from system import create_example_system

"""
对比基准：实数编码遗传算法 + 罚函数（Reviewer 1 第 15 条的基准 (c)）

与 tpe.py 使用同一套三级评价（几何预检 → PG-DNN 粗筛 → Hertz 求解器精筛），
评价次数预算相同（默认 种群 40 × 25 代 = 1000 次）。
适应度 = O_b + R · Σ max(0, c_k / s_k)，几何不可解或接触量无法计算时取 1e5。

用法：
    python ga_baseline.py --seed 42
"""

# 约束归一化尺度：c1 无量纲，c2/c3 为长度 [mm]，c4 为应力 [MPa]
SCALE = np.array([1.0, 100.0, 100.0, 1500.0])
FAILED = 1e5


def fitness(res, penalty):
    if res['stage'] == 'pruned_geometry' or np.isnan(res['value']):
        return FAILED
    violation = np.maximum(0.0, np.array(res['constraints']) / SCALE).sum()
    return res['value'] + penalty * violation


def sbx(p1, p2, low, high, rng, eta=15.0):
    """模拟二进制交叉"""
    u = rng.random(p1.size)
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return np.clip(c1, low, high), np.clip(c2, low, high)


def mutate(x, low, high, rng, pm, eta=20.0):
    """多项式变异"""
    y = x.copy()
    for i in range(x.size):
        if rng.random() < pm:
            u = rng.random()
            delta = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            y[i] = np.clip(x[i] + delta * (high[i] - low[i]), low[i], high[i])
    return y


def main():
    parser = argparse.ArgumentParser(description='Genetic algorithm with penalty function (baseline)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pop', type=int, default=40)
    parser.add_argument('--generations', type=int, default=25)
    parser.add_argument('--penalty', type=float, default=1000.0, help='penalty factor R [MPa per unit violation]')
    parser.add_argument('--crossover', type=float, default=0.9)
    parser.add_argument('--elite', type=int, default=2)
    parser.add_argument('--surrogate', choices=['pinn', 'solver'], default='pinn')
    parser.add_argument('--width-factor', type=float, default=0.8)
    parser.add_argument('--p0', type=float, default=1347.76)  # sigma_HP by ISO 6336-2:2019 (KISSsoft 2026)
    parser.add_argument('--bounds', choices=list(BOUNDS), default='paper')
    parser.add_argument('--beta3-min', type=float, default=5.0)
    parser.add_argument('--beta3-max', type=float, default=30.0)
    parser.add_argument('--model-dir', default='pgnn_model')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    args.out = args.out or os.path.join('results_ga', f"seed{args.seed}")
    os.makedirs(args.out, exist_ok=True)
    gear_env_v2.PINN_MODEL_DIR = args.model_dir

    rng = np.random.default_rng(args.seed)
    bounds = get_bounds(args)
    low = np.array([bounds[k][0] for k in VAR_NAMES])
    high = np.array([bounds[k][1] for k in VAR_NAMES])
    base_params = create_example_system()

    records = []

    def evaluate(x, generation):
        res = evaluate_design(dict(zip(VAR_NAMES, map(float, x))), args, base_params)
        f = fitness(res, args.penalty)
        m = res['metrics']
        rec = {'number': len(records) + 1, 'generation': generation, 'stage': res['stage'],
               **dict(zip(VAR_NAMES, map(float, x))), 'value': res['value'], 'fitness': f,
               **{f'c{i + 1}': c for i, c in enumerate(res['constraints'])}}
        for k in ['min_safety', 'kesai_L', 'kesai_R', 'Pmax_pair1', 'a_len_pair1', 'Pmax_pair2', 'a_len_pair2']:
            rec[k] = m.get(k, '')
        records.append(rec)
        return f

    start = time.time()
    pop = low + rng.random((args.pop, low.size)) * (high - low)
    fit = np.array([evaluate(x, 0) for x in pop])
    for g in range(1, args.generations):
        order = np.argsort(fit)
        children = [pop[i].copy() for i in order[:args.elite]]
        child_fit = [fit[i] for i in order[:args.elite]]
        while len(children) < args.pop:
            # 二元锦标赛选择
            a, b = rng.integers(0, args.pop, 2), rng.integers(0, args.pop, 2)
            p1 = pop[a[np.argmin(fit[a])]]
            p2 = pop[b[np.argmin(fit[b])]]
            c1, c2 = sbx(p1, p2, low, high, rng) if rng.random() < args.crossover else (p1.copy(), p2.copy())
            for c in (c1, c2):
                if len(children) < args.pop:
                    c = mutate(c, low, high, rng, pm=1.0 / low.size)
                    children.append(c)
                    child_fit.append(evaluate(c, g))
        pop, fit = np.array(children), np.array(child_fit)
        print(f"generation {g}: best fitness {fit.min():.2f}")
    duration = time.time() - start

    with open(os.path.join(args.out, 'ga_results.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    feasible = [r for r in records if r['stage'] == 'solver_feasible']
    best = min(feasible, key=lambda r: r['value']) if feasible else None
    stages = [r['stage'] for r in records]
    summary = {'settings': vars(args), 'wall_clock_s': duration, 'n_evaluations': len(records),
               'stage_counts': {s: stages.count(s) for s in sorted(set(stages))},
               'first_feasible_evaluation': feasible[0]['number'] if feasible else None,
               'timing': STATS, 'best': best}
    with open(os.path.join(args.out, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print(json.dumps({k: summary[k] for k in ['wall_clock_s', 'stage_counts', 'first_feasible_evaluation']},
                     ensure_ascii=False))
    print('best:', best)


if __name__ == "__main__":
    main()
