"""
用修正后的 Hertz 求解器重新计算训练数据（v9）

沿用原数据文件中的 LHS 输入（12 个特征），只重新计算 Pmax 和 a_len，
因此采样设计与原数据完全相同。求解器判为无效的样本（线接触、接触椭圆过于细长）被剔除。

用法：
    python regenerate_training_data.py --src pinn_training_data.csv --dst pinn_training_data_v9.csv --workers 10
"""

import argparse
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

FEATURES = ['gama1', 'gama2', 'beta1', 'beta2', 'N1', 'N2', 'mn', 'Fn', 'kesai', 'an', 'E', 'nu']


def solve_row(row):
    from contact_stress import calculate_hertz_contact_stress
    p = {
        'gama1': np.radians(row['gama1']), 'gama2': np.radians(row['gama2']),
        'beta1': np.radians(row['beta1']), 'beta2': np.radians(row['beta2']),
        'N1': int(row['N1']), 'N2': int(row['N2']), 'mn': row['mn'],
        'an': np.radians(row['an']), 'E': row['E'], 'nu': row['nu'],
        'b1': 188.0, 'b2': 180.0, 'Fn': row['Fn'], 'kesai': row['kesai'],
        'sigma_rad': np.radians(row['kesai']), 'xt1': 0.0, 'xt2': 0.0, 'fuhao': -1, 'drive_gear': 1,
    }
    try:
        r = calculate_hertz_contact_stress(p)
    except Exception as e:  # 记录失败原因，不中断整批计算
        return {'valid': False, 'error': str(e)}
    if not r.get('valid', False):
        return {'valid': False, 'error': r.get('error', '')}
    return {'valid': True, 'Pmax': r['Pmax'], 'a_len': r['a_len']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='pinn_training_data.csv')
    parser.add_argument('--dst', default='pinn_training_data_v9.csv')
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()

    src = pd.read_csv(args.src)
    for c in FEATURES:
        src[c] = pd.to_numeric(src[c], errors='coerce')
    src = src.dropna(subset=FEATURES).reset_index(drop=True)
    rows = src[FEATURES].to_dict('records')
    print(f"输入样本 {len(rows)} 个，开始计算……")

    t0 = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(solve_row, rows, chunksize=20)
    print(f"计算耗时 {time.time() - t0:.1f} s")

    out = []
    errors = {}
    for row, res in zip(rows, results):
        if res['valid']:
            out.append({**row, 'Pmax': res['Pmax'], 'a_len': res['a_len']})
        else:
            errors[res['error']] = errors.get(res['error'], 0) + 1
    df = pd.DataFrame(out)
    df.to_csv(args.dst, index=False)
    print(f"有效样本 {len(df)} 个，已保存至 {args.dst}")
    print("无效原因：", errors)


if __name__ == '__main__':
    main()
