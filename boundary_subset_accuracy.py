"""
测试集中“边界样本”的精度（审稿意见 R1-7）

边界样本：归一化输入空间中离中心最远的 10% 测试样本（欧氏距离，范围按全部有效样本计算）。
输入为 12 维，“任一输入位于范围两端各 10% 以内”会把约 94% 的样本划为边界样本，因此不采用。
离散输入（齿数）也按同样规则处理。

用法：
    python boundary_subset_accuracy.py --data data/pgnn_training_data.csv --model-dir pgnn_model
"""

import argparse
import json
import os

import numpy as np

from pgnn_contact_stress import PIINStressPredictor, load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/pgnn_training_data.csv')
    parser.add_argument('--model-dir', default='pgnn_model')
    parser.add_argument('--margin', type=float, default=0.10)
    args = parser.parse_args()

    df, _ = load_dataset(args.data)
    names = PIINStressPredictor.FEATURE_NAMES
    X = df[names].values.astype(np.float64)
    y = df[PIINStressPredictor.OUTPUT_NAMES].values.astype(np.float64)
    te = np.load(os.path.join(args.model_dir, 'split_indices.npz'))['test']

    lo, hi = X.min(axis=0), X.max(axis=0)
    width = np.where(hi > lo, hi - lo, 1.0)
    rel = (X[te] - lo) / width
    dist = np.linalg.norm(rel - 0.5, axis=1)
    boundary = dist >= np.quantile(dist, 1 - args.margin)

    model = PIINStressPredictor()
    model.load(args.model_dir)
    out = {'margin': args.margin, 'n_test': int(len(te)),
           'n_boundary': int(boundary.sum()), 'n_interior': int((~boundary).sum())}
    for tag, mask in (('boundary', boundary), ('interior', ~boundary)):
        res = model.evaluate(X[te][mask], y[te][mask])
        out[tag] = res
        print(f"[{tag}] n={mask.sum()}  " + ", ".join(f"{k}={v:.4g}" for k, v in res.items()))
    with open(os.path.join(args.model_dir, 'boundary_subset.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
