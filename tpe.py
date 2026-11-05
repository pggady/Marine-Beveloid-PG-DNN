# 先导入PyTorch，确保在其他库之前加载（避免DLL冲突）
import torch
import optuna
import numpy as np
import argparse
import csv
import json
import os
import platform
import time
from datetime import datetime

# 引入环境定义
import gear_env_v2
from gear_env_v2 import GearEnvV2
from gear_calculations import GearCalculations
from system import create_example_system

"""
PG-DNN 加速的约束 TPE 优化（与论文 Eq.(35)–(36) 及 Algorithm 1 一致）

目标：  min O_b = p13 + p23
约束：  c1 = 1.25 - S_min                      <= 0
        c2 = 2 a13 - w * min(b1, b3)          <= 0
        c3 = 2 a23 - w * min(b2, b3)          <= 0
        c4 = max(p13, p23) - p0               <= 0
其中 pair1 = 相交轴（齿轮3 + 齿轮2）→ p23, a23；pair2 = 交错轴（齿轮3 + 齿轮1）→ p13, a13

三级评价：
    1. 几何预检：节锥方程无解 → 剪枝
    2. PG-DNN 预测 p、a，计算 c1–c4
    3. PG-DNN 判定全部满足约束 → 调用 Hertz 求解器复算 p、a，更新约束和目标
约束通过 TPESampler 的 constraints_func 传入（不使用罚函数）

用法示例：
    python tpe.py                                   # 默认：PG-DNN 粗筛 + 求解器精筛，种子 42
    python tpe.py --seed 1 --out results_tpe/seed1  # 不同随机种子
    python tpe.py --surrogate solver                # 基准：每个几何可行的候选都用 Hertz 求解器
    python tpe.py --width-factor 0.75               # 0.8 系数的敏感性分析
"""

# 搜索范围：键为代码中的参数名，括号内为论文符号
BOUNDS = {
    'paper': {'gama1': (4.0, 6.0),        # γ3
              'gama2_gear1': (0.0, 2.5),  # γ1
              'gama2_gear2': (0.5, 3.5),  # γ2
              'xt3': (-0.5, 0.5)},        # x_t3
    'code_v8': {'gama1': (4.0, 6.0),
                'gama2_gear1': (0.0, 3.5),
                'gama2_gear2': (0.0, 3.5),
                'xt3': (-0.5, 0.5)},
}
SAFETY_TARGET = 1.25
INFEASIBLE = [1e3, 1e3, 1e3, 1e3]  # 剪枝试验的约束值（仅用于排序，不进入目标函数）

STATS = {'geometry_s': 0.0, 'solver_s': 0.0, 'solver_calls': 0, 'pinn_calls': 0}


def build_constraints(m, args, b):
    """按论文 Eq.(35) 计算 c1–c4；接触量缺失时视为不满足"""
    p23, a23 = m.get('Pmax_pair1', np.nan), m.get('a_len_pair1', np.nan)
    p13, a13 = m.get('Pmax_pair2', np.nan), m.get('a_len_pair2', np.nan)
    if any(np.isnan(v) for v in (p13, a13, p23, a23)):
        return [SAFETY_TARGET - m.get('min_safety', 0.0), 1e3, 1e3, 1e3], np.nan
    c = [SAFETY_TARGET - m['min_safety'],
         2 * a13 - args.width_factor * min(b['b1'], b['b3']),
         2 * a23 - args.width_factor * min(b['b2'], b['b3']),
         max(p13, p23) - args.p0]
    return c, p13 + p23


def get_bounds(args):
    """决策变量的搜索范围"""
    bounds = dict(BOUNDS[args.bounds])
    bounds['beta3'] = (args.beta3_min, args.beta3_max)
    return bounds


VAR_NAMES = ['gama1', 'gama2_gear1', 'gama2_gear2', 'beta3', 'xt3']


def evaluate_design(params, args, base_params=None):
    """
    三级评价一个设计（TPE、遗传算法基准、鲁棒性分析共用）

    Returns:
        dict: stage, value (O_b，无法计算时为 nan), constraints, metrics（含 *_pinn 字段）
    """
    base_params = base_params or create_example_system()
    widths = {'b1': base_params['gear1']['b2'], 'b2': base_params['gear2']['b2'], 'b3': base_params['gear3']['b1']}

    # 1. 实例化环境并注入参数
    env = GearEnvV2(base_params)
    env.current_params.update(params)

    # 2. 几何预检 + PG-DNN 预测
    t0 = time.time()
    m = env._calculate_metrics()
    STATS['geometry_s'] += time.time() - t0
    if not m.get('valid', False):
        return {'stage': 'pruned_geometry', 'value': np.nan, 'constraints': INFEASIBLE, 'metrics': m}
    STATS['pinn_calls'] += 2

    constraints, value = build_constraints(m, args, widths)
    m['constraints_pinn'] = constraints
    for k in ['Pmax_pair1', 'a_len_pair1', 'Pmax_pair2', 'a_len_pair2']:
        m[k + '_pinn'] = float(m.get(k, np.nan))

    # 3. 精筛：PG-DNN 判定可行（或基准模式）时调用 Hertz 求解器
    if args.surrogate == 'solver' or all(c <= 0 for c in constraints):
        t0 = time.time()
        m.update(env.evaluate_contact_solver(m))
        STATS['solver_s'] += time.time() - t0
        STATS['solver_calls'] += 2
        constraints, value = build_constraints(m, args, widths)
        stage = 'solver_feasible' if all(c <= 0 for c in constraints) else 'solver_infeasible'
    else:
        stage = 'surrogate_infeasible'
    return {'stage': stage, 'value': value, 'constraints': [float(c) for c in constraints], 'metrics': m}


def make_objective(args):
    base_params = create_example_system()
    bounds = get_bounds(args)

    def objective(trial):
        # 1. 定义搜索空间
        params = {k: trial.suggest_float(k, *bounds[k]) for k in VAR_NAMES}

        # 2. 三级评价
        res = evaluate_design(params, args, base_params)
        trial.set_user_attr('stage', res['stage'])
        trial.set_user_attr('constraints', res['constraints'])
        if res['stage'] == 'pruned_geometry':
            raise optuna.exceptions.TrialPruned()

        # 3. 记录指标
        m = res['metrics']
        trial.set_user_attr('constraints_pinn', [float(c) for c in m['constraints_pinn']])
        for k in ['min_safety', 'kesai_L', 'kesai_R', 'Pmax_pair1', 'a_len_pair1', 'Pmax_pair2', 'a_len_pair2',
                  'Pmax_pair1_pinn', 'a_len_pair1_pinn', 'Pmax_pair2_pinn', 'a_len_pair2_pinn']:
            trial.set_user_attr(k, float(m.get(k, np.nan)))
        value = res['value']
        if np.isnan(value):
            value = 1e4  # 接触量无法计算：目标取大值，同时约束已标为不满足
        return float(value)

    return objective


def save_results(study, args, duration, path_csv, path_json):
    """导出每个试验的数据和汇总信息"""
    fields = ['number', 'state', 'stage', 'gama1', 'gama2_gear1', 'gama2_gear2', 'beta3', 'xt3', 'value',
              'min_safety', 'kesai_L', 'kesai_R', 'Pmax_pair1', 'a_len_pair1', 'Pmax_pair2', 'a_len_pair2',
              'Pmax_pair1_pinn', 'a_len_pair1_pinn', 'Pmax_pair2_pinn', 'a_len_pair2_pinn',
              'c1', 'c2', 'c3', 'c4']
    with open(path_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in study.trials:
            row = {'number': t.number + 1, 'state': t.state.name, 'stage': t.user_attrs.get('stage', ''),
                   'value': t.value if t.value is not None else ''}
            row.update({k: t.params.get(k, '') for k in ['gama1', 'gama2_gear1', 'gama2_gear2', 'beta3', 'xt3']})
            row.update({k: t.user_attrs.get(k, '') for k in fields if k in t.user_attrs})
            cons = t.user_attrs.get('constraints', [''] * 4)
            row.update({f'c{i + 1}': cons[i] for i in range(4)})
            writer.writerow(row)

    stages = [t.user_attrs.get('stage', '') for t in study.trials]
    feasible = [t for t in study.trials if t.user_attrs.get('stage') == 'solver_feasible']
    best = min(feasible, key=lambda t: t.value) if feasible else None
    summary = {
        'settings': vars(args),
        'wall_clock_s': duration,
        'n_trials': len(study.trials),
        'stage_counts': {s: stages.count(s) for s in sorted(set(stages))},
        'first_feasible_trial': (min(t.number for t in feasible) + 1) if feasible else None,
        'timing': STATS,
        'matlab_restarts': GearCalculations.restarts,
        'best': None if best is None else {
            'trial': best.number + 1, 'O_b': best.value, 'params': best.params,
            **{k: best.user_attrs.get(k) for k in ['min_safety', 'kesai_L', 'kesai_R', 'Pmax_pair1',
                                                     'a_len_pair1', 'Pmax_pair2', 'a_len_pair2', 'constraints']}},
        'versions': {'python': platform.python_version(), 'optuna': optuna.__version__, 'torch': torch.__version__,
                     'numpy': np.__version__},
        'pinn_model_dir': gear_env_v2.PINN_MODEL_DIR,
        'finished': datetime.now().isoformat(timespec='seconds'),
    }
    with open(path_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def run_optimization(args):
    os.makedirs(args.out, exist_ok=True)
    gear_env_v2.PINN_MODEL_DIR = args.model_dir

    print("=" * 50)
    print(f"开始约束 TPE 优化：seed={args.seed}, trials={args.n_trials}, surrogate={args.surrogate}, "
          f"width_factor={args.width_factor}, p0={args.p0}")
    print("=" * 50)

    # 1. 创建 Study（约束通过 constraints_func 传入）
    sampler = optuna.samplers.TPESampler(
        seed=args.seed, n_startup_trials=args.n_startup,
        constraints_func=lambda t: t.user_attrs.get('constraints', INFEASIBLE))
    # 计时运行用内存存储（SQLite 每次采样都要读回全部历史，会增加耗时）
    storage = None if args.storage == 'memory' else f"sqlite:///{os.path.join(args.out, 'study.db')}"
    study = optuna.create_study(direction='minimize', sampler=sampler, storage=storage,
                                study_name=f"seed{args.seed}_{args.surrogate}", load_if_exists=False)

    start = time.time()
    study.optimize(make_objective(args), n_trials=args.n_trials, show_progress_bar=True)
    duration = time.time() - start

    summary = save_results(study, args, duration, os.path.join(args.out, 'optuna_results.csv'),
                           os.path.join(args.out, 'summary.json'))
    print("\n" + "=" * 50)
    print(f"🎉 优化完成！总耗时 {duration:.2f} 秒")
    print("各阶段试验数：", summary['stage_counts'])
    if summary['best']:
        b = summary['best']
        print(f"最优可行解（求解器验证）: trial {b['trial']}, O_b = {b['O_b']:.2f} MPa")
        print(f"  p23 = {b['Pmax_pair1']:.2f} MPa, a23 = {b['a_len_pair1']:.2f} mm, "
              f"p13 = {b['Pmax_pair2']:.2f} MPa, a13 = {b['a_len_pair2']:.2f} mm, S_min = {b['min_safety']:.4f}")
        for k, v in b['params'].items():
            print(f"  {k}: {v:.6f}")
    else:
        print("未找到经求解器验证的可行解")
    print("=" * 50)


def parse_args():
    parser = argparse.ArgumentParser(description='PG-DNN accelerated constrained TPE optimization')
    parser.add_argument('--n-trials', type=int, default=1000)
    parser.add_argument('--n-startup', type=int, default=10, help='random start-up trials of the TPE')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--surrogate', choices=['pinn', 'solver'], default='pinn',
                        help='pinn: PG-DNN screening + solver verification; solver: solver for every trial')
    parser.add_argument('--width-factor', type=float, default=0.8, help='factor of the face-width constraint')
    parser.add_argument('--p0', type=float, default=1347.76, help='permissible contact stress sigma_HP by ISO 6336-2:2019, KISSsoft 2026 [MPa]')
    parser.add_argument('--bounds', choices=list(BOUNDS), default='paper')
    parser.add_argument('--beta3-min', type=float, default=5.0)
    parser.add_argument('--beta3-max', type=float, default=30.0)
    parser.add_argument('--model-dir', default='pgnn_model')
    parser.add_argument('--out', default=None)
    parser.add_argument('--storage', choices=['sqlite', 'memory'], default='sqlite')
    args = parser.parse_args()
    if args.out is None:
        args.out = os.path.join('results_tpe', f"{args.surrogate}_w{args.width_factor}_seed{args.seed}")
    return args


if __name__ == "__main__":
    run_optimization(parse_args())
