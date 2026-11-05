"""
生成 PG-DNN 训练数据（LHS 采样 + 修正后的 Hertz 求解器，多进程）

β3 的搜索上界放宽到 30° 后，交错副小轮螺旋角可达约 −34°，相交副 FPD 角可达约 −8°，
因此螺旋角范围取 ±36°、FPD 角取 ±12°，其余范围与原数据相同。求解器判为无效的样本被剔除。

用法：
    python generate_lhs_dataset.py --n 10500 --seed 42 --dst data/pgnn_training_data.csv --workers 10
"""

import argparse
import time
from multiprocessing import Pool

import pandas as pd
from scipy.stats import qmc

from regenerate_training_data import FEATURES, solve_row

PARAM_RANGES = {
    'gama1': (0.1, 5.0),
    'gama2': (2.0, 8.0),
    'beta1': (-36.0, 36.0),
    'beta2': (-36.0, 36.0),
    'N1': (17, 40),
    'N2': (40, 90),
    'mn': (2.0, 8.0),
    'Fn': (10000.0, 100000.0),
    'kesai': (-12.0, 12.0),
    'an': (18.0, 25.0),
    'E': (180000.0, 220000.0),
    'nu': (0.2, 0.4),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dst', default='data/pgnn_training_data.csv')
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()

    samples = qmc.LatinHypercube(d=len(FEATURES), seed=args.seed).random(n=args.n)
    rows = []
    for s in samples:
        row = {}
        for j, name in enumerate(FEATURES):
            low, high = PARAM_RANGES[name]
            v = low + s[j] * (high - low)
            row[name] = int(v) if name in ('N1', 'N2') else v
        rows.append(row)
    print(f"LHS 样本 {len(rows)} 个，开始计算……")

    t0 = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(solve_row, rows, chunksize=20)
    print(f"计算耗时 {time.time() - t0:.1f} s")

    out, errors = [], {}
    for row, res in zip(rows, results):
        if res['valid']:
            out.append({**row, 'Pmax': res['Pmax'], 'a_len': res['a_len']})
        else:
            errors[res['error']] = errors.get(res['error'], 0) + 1
    pd.DataFrame(out).to_csv(args.dst, index=False)
    print(f"有效样本 {len(out)} 个，已保存至 {args.dst}")
    print("无效原因：", errors)


if __name__ == '__main__':
    main()
