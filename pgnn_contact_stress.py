"""
物理引导神经网络(PG-DNN) - 接触应力预测模型
预测齿轮副的最大接触应力 Pmax 与接触椭圆半长轴 a_len

v9 修改：
1. 物理损失按特征名取索引（Fn、kesai），不再写死数字索引
2. 数据按 训练/验证/测试 = 70/15/15 划分，标准化参数只用训练集计算
3. 损失写成 L = L_MSE + λ1·L_Pos + λ2·L_Load + λ3·L_FPD，三个权重可单独设置
4. 记录各损失分项、梯度范数、物理约束违反率；测试集报告 MAE/RMSE/MAPE/R²
5. 支持 λ 网格搜索与 λ=0 消融对比

用法示例：
    python pgnn_contact_stress.py --lambdas 0.01 0.001 0.001          # 训练单个模型
    python pgnn_contact_stress.py --grid --grid-epochs 300            # 网格搜索 λ，再用最优 λ 完整训练
    python pgnn_contact_stress.py --ablation                          # 额外训练 λ=0 的对照模型
"""

import argparse
import itertools
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContactStressPINN(nn.Module):
    """
    物理引导神经网络 - 接触应力预测

    输入特征 (12维):
        gama1, gama2, beta1, beta2, N1, N2, mn, Fn, kesai, an, E, nu

    输出 (2维): Pmax (MPa), a_len (mm)
    """

    def __init__(self, input_dim=12, output_dim=2):
        super(ContactStressPINN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(64),

            nn.Linear(64, 128),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(128),

            nn.Linear(128, 256),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(256),

            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(256),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),

            nn.Linear(64, output_dim)
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class PIINStressPredictor:
    """
    PG-DNN 接触应力预测器 - 管理模型训练、评估和推理
    """

    # 输入特征顺序 (12维) - 不含变位系数，计算时变位系数取0
    FEATURE_NAMES = [
        'gama1', 'gama2', 'beta1', 'beta2', 'N1', 'N2', 'mn',
        'Fn', 'kesai', 'an', 'E', 'nu'
    ]
    # 物理约束用到的特征索引按名字确定
    IDX_FN = FEATURE_NAMES.index('Fn')
    IDX_KESAI = FEATURE_NAMES.index('kesai')

    # 输出特征顺序 (2维)
    OUTPUT_NAMES = ['Pmax', 'a_len']

    # FPD 角绝对值小于该值(度)的样本不参与 FPD 违反率统计（导数理论上为0，符号无意义）
    KESAI_EPS = 0.1

    def __init__(self, lambda_pos=0.01, lambda_load=0.001, lambda_fpd=0.001,
                 w_a=2.5, learning_rate=1e-3, weight_decay=1e-5, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = ContactStressPINN(input_dim=12, output_dim=2).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # 数据标准化参数（只由训练集计算）
        self.scaler_mean = None
        self.scaler_scale = None
        self.output_mean = None
        self.output_scale = None

        # 损失权重
        self.lambda_pos = lambda_pos
        self.lambda_load = lambda_load
        self.lambda_fpd = lambda_fpd
        self.w_a = w_a
        self.history = []

    # ------------------------------------------------------------------ 标准化
    def fit_scaler(self, X, y):
        """计算标准化参数（只传入训练集）"""
        self.scaler_mean = np.mean(X, axis=0)
        self.scaler_scale = np.std(X, axis=0) + 1e-8
        self.output_mean = np.mean(y, axis=0)
        self.output_scale = np.std(y, axis=0) + 1e-8

    def transform_input(self, X):
        """标准化输入"""
        return (X - self.scaler_mean) / self.scaler_scale

    def transform_output(self, y):
        """标准化输出"""
        return (y - self.output_mean) / self.output_scale

    def inverse_transform_output(self, y_scaled):
        """反标准化输出"""
        if isinstance(y_scaled, torch.Tensor):
            scale = torch.tensor(self.output_scale, device=y_scaled.device, dtype=y_scaled.dtype)
            mean = torch.tensor(self.output_mean, device=y_scaled.device, dtype=y_scaled.dtype)
            return y_scaled * scale + mean
        return y_scaled * self.output_scale + self.output_mean

    # ------------------------------------------------------------------ 物理损失
    def _input_gradients(self, X_scaled, y_pred):
        """标准化坐标下 Pmax、a_len 对输入的梯度（保留计算图，用于二阶反传）"""
        grad_p = torch.autograd.grad(y_pred[:, 0].sum(), X_scaled, create_graph=True, retain_graph=True)[0]
        grad_a = torch.autograd.grad(y_pred[:, 1].sum(), X_scaled, create_graph=True, retain_graph=True)[0]
        return grad_p, grad_a

    def physics_terms(self, X_scaled, y_pred):
        """
        返回三个物理损失项（均为批平均）：
            L_Pos  = mean[ReLU(-Pmax) + ReLU(-a_len)]                （物理单位）
            L_Load = mean[ReLU(-dPmax/dFn) + ReLU(-da_len/dFn)]       （标准化坐标）
            L_FPD  = mean[ReLU(-kesai · dPmax/dkesai)]                （kesai 取物理值的符号）
        """
        y_real = self.inverse_transform_output(y_pred)
        l_pos = torch.mean(torch.relu(-y_real[:, 0]) + torch.relu(-y_real[:, 1]))

        grad_p, grad_a = self._input_gradients(X_scaled, y_pred)
        l_load = torch.mean(torch.relu(-grad_p[:, self.IDX_FN]) + torch.relu(-grad_a[:, self.IDX_FN]))

        kesai_mean = float(self.scaler_mean[self.IDX_KESAI])
        kesai_scale = float(self.scaler_scale[self.IDX_KESAI])
        kesai_real = X_scaled[:, self.IDX_KESAI].detach() * kesai_scale + kesai_mean
        l_fpd = torch.mean(torch.relu(-kesai_real * grad_p[:, self.IDX_KESAI]))
        return l_pos, l_load, l_fpd

    # ------------------------------------------------------------------ 训练
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=2000, batch_size=128,
              eval_every=50, verbose=True, lr_schedule='constant', lr_min=1e-5):
        """训练模型；标准化参数只用训练集；按验证集 MSE 保留最优权重
        lr_schedule='cosine' 时学习率按余弦规律从初值降到 lr_min"""
        self.fit_scaler(X_train, y_train)
        X_scaled = self.transform_input(X_train).astype(np.float32)
        y_scaled = self.transform_output(y_train).astype(np.float32)

        dataloader = DataLoader(TensorDataset(torch.tensor(X_scaled), torch.tensor(y_scaled)),
                                batch_size=batch_size, shuffle=True)
        mse_loss = nn.MSELoss()
        use_physics = bool((self.lambda_pos > 0) or (self.lambda_load > 0) or (self.lambda_fpd > 0))

        best_val, best_state = np.inf, None
        self.history = []
        scheduler = (optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=lr_min)
                     if lr_schedule == 'cosine' else None)
        for epoch in range(epochs):
            self.model.train()
            sums = dict(data=0.0, pos=0.0, load=0.0, fpd=0.0, grad_norm=0.0)
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device).requires_grad_(use_physics)
                batch_y = batch_y.to(device)
                self.optimizer.zero_grad()

                y_pred = self.model(batch_X)
                data_loss = mse_loss(y_pred[:, 0], batch_y[:, 0]) + self.w_a * mse_loss(y_pred[:, 1], batch_y[:, 1])
                total_loss = data_loss
                if use_physics:
                    l_pos, l_load, l_fpd = self.physics_terms(batch_X, y_pred)
                    total_loss = total_loss + self.lambda_pos * l_pos + self.lambda_load * l_load + self.lambda_fpd * l_fpd
                    sums['pos'] += l_pos.item()
                    sums['load'] += l_load.item()
                    sums['fpd'] += l_fpd.item()

                total_loss.backward()
                # 梯度裁剪：导数惩罚需要二阶自动微分，LeakyReLU 网络的输入导数为分段常数
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                sums['data'] += data_loss.item()
                sums['grad_norm'] += float(grad_norm)

            record = {k: v / len(dataloader) for k, v in sums.items()}
            record['epoch'] = epoch + 1
            record['lr'] = self.optimizer.param_groups[0]['lr']
            if scheduler is not None:
                scheduler.step()
            if X_val is not None and ((epoch + 1) % eval_every == 0 or epoch + 1 == epochs):
                val = self.evaluate(X_val, y_val)
                record.update({'val_mse_scaled': val['mse_scaled'],
                               'val_viol_load': val['viol_load_p'],
                               'val_viol_fpd': val['viol_fpd']})
                if val['mse_scaled'] < best_val:
                    best_val = val['mse_scaled']
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - Data: {record['data']:.6f}, Pos: {record['pos']:.2e}, "
                          f"Load: {record['load']:.2e}, FPD: {record['fpd']:.2e}, Val MSE: {val['mse_scaled']:.6f}, "
                          f"Viol load/FPD: {val['viol_load_p']:.4f}/{val['viol_fpd']:.4f}")
            self.history.append(record)

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.history

    # ------------------------------------------------------------------ 评估
    def evaluate(self, X, y):
        """精度指标（物理单位）与物理约束违反率（标准化坐标下的输入导数）"""
        X_scaled = torch.tensor(self.transform_input(X).astype(np.float32), device=device, requires_grad=True)
        self.model.eval()
        y_pred = self.model(X_scaled)
        grad_p, grad_a = self._input_gradients(X_scaled, y_pred)

        pred = self.inverse_transform_output(y_pred.detach().cpu().numpy())
        true = np.asarray(y, dtype=np.float64)
        err = pred - true
        out = {'mse_scaled': float(np.mean(((pred - true) / self.output_scale) ** 2 * np.array([1.0, self.w_a])))}
        for k, name in enumerate(self.OUTPUT_NAMES):
            ss_res = np.sum(err[:, k] ** 2)
            ss_tot = np.sum((true[:, k] - true[:, k].mean()) ** 2)
            out[f'{name}_MAE'] = float(np.mean(np.abs(err[:, k])))
            out[f'{name}_RMSE'] = float(np.sqrt(np.mean(err[:, k] ** 2)))
            out[f'{name}_MAPE'] = float(np.mean(np.abs(err[:, k]) / np.abs(true[:, k])) * 100)
            out[f'{name}_R2'] = float(1 - ss_res / ss_tot)

        g_p = grad_p.detach().cpu().numpy()
        g_a = grad_a.detach().cpu().numpy()
        kesai = X[:, self.IDX_KESAI]
        mask = np.abs(kesai) > self.KESAI_EPS
        out['viol_load_p'] = float(np.mean(g_p[:, self.IDX_FN] < 0))
        out['viol_load_a'] = float(np.mean(g_a[:, self.IDX_FN] < 0))
        out['viol_fpd'] = float(np.mean(kesai[mask] * g_p[mask, self.IDX_KESAI] < 0))
        out['viol_pos'] = float(np.mean((pred[:, 0] <= 0) | (pred[:, 1] <= 0)))
        return out

    # ------------------------------------------------------------------ 推理
    def predict(self, X):
        """
        预测接触应力

        Args:
            X: numpy array, shape (N, 12) 或 dict

        Returns:
            numpy array, shape (N, 2): [Pmax, a_len]
        """
        if isinstance(X, dict):
            X = self._dict_to_array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_tensor = torch.tensor(self.transform_input(X).astype(np.float32)).to(device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor)
        return self.inverse_transform_output(y_pred.cpu().numpy())

    def _dict_to_array(self, params):
        """将参数字典转换为数组"""
        return np.array([params.get(name, 0) for name in self.FEATURE_NAMES], dtype=np.float64)

    # ------------------------------------------------------------------ 保存与加载
    def save(self, save_dir='pgnn_model', extra=None):
        """保存模型、标准化参数和训练信息"""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'model.pth'))
        joblib.dump({
            'scaler_mean': self.scaler_mean,
            'scaler_scale': self.scaler_scale,
            'output_mean': self.output_mean,
            'output_scale': self.output_scale
        }, os.path.join(save_dir, 'scaler.pkl'))
        meta = {'lambda_pos': self.lambda_pos, 'lambda_load': self.lambda_load,
                'lambda_fpd': self.lambda_fpd, 'w_a': self.w_a}
        if extra:
            meta.update(extra)
        with open(os.path.join(save_dir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        pd.DataFrame(self.history).to_csv(os.path.join(save_dir, 'history.csv'), index=False)
        print(f"模型已保存至 {save_dir}")

    def load(self, save_dir='pgnn_model'):
        """加载模型"""
        self.model.load_state_dict(
            torch.load(os.path.join(save_dir, 'model.pth'), map_location=device)
        )
        data = joblib.load(os.path.join(save_dir, 'scaler.pkl'))
        self.scaler_mean = data['scaler_mean']
        self.scaler_scale = data['scaler_scale']
        self.output_mean = data['output_mean']
        self.output_scale = data['output_scale']
        print("模型加载成功")


def generate_training_data(n_samples=2000, save_path='data/pgnn_training_data.csv'):
    """
    生成PG-DNN训练数据（LHS 采样 + Hertz 求解器）
    """
    from contact_stress import calculate_hertz_contact_stress
    from scipy.stats import qmc

    print(f"开始生成 {n_samples} 个训练样本...")

    # 参数范围 (12个输入特征，不含变位系数)
    param_ranges = {
        'gama1': (0.1, 5.0),
        'gama2': (2, 8.0),
        'beta1': (-30.0, 30.0),
        'beta2': (-30.0, 30.0),
        'N1': (17, 40),
        'N2': (40, 90),
        'mn': (2, 8),
        'Fn': (10000, 100000),
        'kesai': (-10.0, 10.0),
        'an': (18.0, 25.0),
        'E': (180000, 220000),
        'nu': (0.2, 0.4),
    }
    fixed_params = {'b1': 188, 'b2': 180, 'fuhao': -1}

    samples = qmc.LatinHypercube(d=len(param_ranges)).random(n=n_samples)
    data = []
    for i, sample in enumerate(samples):
        params = fixed_params.copy()
        for j, (name, (low, high)) in enumerate(param_ranges.items()):
            params[name] = int(low + sample[j] * (high - low)) if name in ['N1', 'N2'] else low + sample[j] * (high - low)

        row = {k: params[k] for k in PIINStressPredictor.FEATURE_NAMES}
        for k in ['gama1', 'gama2', 'beta1', 'beta2', 'an']:
            params[k] = np.radians(params[k])
        params['sigma_rad'] = np.radians(params['kesai'])
        params['xt1'] = 0.0
        params['xt2'] = 0.0
        params['drive_gear'] = 1

        result = calculate_hertz_contact_stress(params)
        if result.get('valid', False) and result.get('Pmax', 0) > 0 and result.get('a_len', 0) > 0:
            row['Pmax'] = result['Pmax']
            row['a_len'] = result['a_len']
            data.append(row)
        if (i + 1) % 200 == 0:
            print(f"  进度: {i + 1}/{n_samples}, 有效样本: {len(data)}")

    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"数据生成完成: {len(df)} 个有效样本，保存至 {save_path}")
    return df


def load_dataset(data_path='data/pgnn_training_data.csv'):
    """读取数据并按论文中的规则剔除非物理样本"""
    df = pd.read_csv(data_path)
    n_raw = len(df)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    df = df[(df['Pmax'] >= 400) & (df['Pmax'] <= 2500) & (df['a_len'] <= 100)].reset_index(drop=True)
    print(f"原始样本 {n_raw} 个，剔除后 {len(df)} 个")
    return df, n_raw


def split_dataset(n, seed=42, ratios=(0.70, 0.15, 0.15)):
    """随机划分 训练/验证/测试 索引"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def main():
    parser = argparse.ArgumentParser(description='Train the physics-guided surrogate of the Hertzian solver')
    parser.add_argument('--data', default='data/pgnn_training_data.csv')
    parser.add_argument('--out', default='pgnn_model')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lambdas', type=float, nargs=3, default=[0.01, 0.001, 0.001],
                        metavar=('L_POS', 'L_LOAD', 'L_FPD'))
    parser.add_argument('--grid', action='store_true', help='grid search of the loss weights on the validation set')
    parser.add_argument('--grid-epochs', type=int, default=300)
    parser.add_argument('--grid-values', type=float, nargs='+', default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    parser.add_argument('--max-violation', type=float, default=0.01,
                        help='largest acceptable fraction of validation samples violating a physical condition')
    parser.add_argument('--ablation', action='store_true', help='also train a model without physical terms')
    parser.add_argument('--lr-schedule', choices=['constant', 'cosine'], default='constant',
                        help='learning-rate schedule of the final training (the grid search always uses constant)')
    args = parser.parse_args()

    df, n_raw = load_dataset(args.data)
    X = df[PIINStressPredictor.FEATURE_NAMES].values.astype(np.float64)
    y = df[PIINStressPredictor.OUTPUT_NAMES].values.astype(np.float64)
    tr, va, te = split_dataset(len(df), seed=args.seed)
    print(f"训练/验证/测试 = {len(tr)}/{len(va)}/{len(te)}")
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, 'split_indices.npz'), train=tr, val=va, test=te)

    lambdas = list(args.lambdas)
    if args.grid:
        # λ1 单独搜索；λ2 与 λ3 取相同值，以控制组合数
        rows = []
        for l_pos, l_phys in itertools.product(args.grid_values, args.grid_values):
            t0 = time.time()
            model = PIINStressPredictor(l_pos, l_phys, l_phys, seed=args.seed)
            model.train(X[tr], y[tr], X[va], y[va], epochs=args.grid_epochs, batch_size=args.batch,
                        eval_every=args.grid_epochs, verbose=False)
            val = model.evaluate(X[va], y[va])
            rows.append({'lambda_pos': l_pos, 'lambda_load': l_phys, 'lambda_fpd': l_phys,
                         'val_mse_scaled': val['mse_scaled'], 'viol_load_p': val['viol_load_p'],
                         'viol_load_a': val['viol_load_a'], 'viol_fpd': val['viol_fpd'],
                         'viol_pos': val['viol_pos'], 'seconds': time.time() - t0})
            print(rows[-1])
        grid = pd.DataFrame(rows)
        grid.to_csv(os.path.join(args.out, 'lambda_grid.csv'), index=False)
        viol = grid[['viol_load_p', 'viol_load_a', 'viol_fpd', 'viol_pos']].max(axis=1)
        ok = grid[viol <= args.max_violation]
        best = (ok if len(ok) else grid.assign(v=viol).sort_values('v').head(5)).sort_values('val_mse_scaled').iloc[0]
        lambdas = [float(best['lambda_pos']), float(best['lambda_load']), float(best['lambda_fpd'])]
        print(f"选定 λ = {lambdas}")

    runs = [('pgdnn', lambdas)]
    if args.ablation:
        runs.append(('no_physics', [0.0, 0.0, 0.0]))

    summary = {'n_raw': n_raw, 'n_filtered': len(df), 'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
               'seed': args.seed, 'epochs': args.epochs, 'batch_size': args.batch, 'torch': torch.__version__,
               'device': str(device), 'lr_schedule': args.lr_schedule}
    for tag, lam in runs:
        t0 = time.time()
        model = PIINStressPredictor(*lam, seed=args.seed)
        model.train(X[tr], y[tr], X[va], y[va], epochs=args.epochs, batch_size=args.batch,
                    lr_schedule=args.lr_schedule)
        train_time = time.time() - t0
        test = model.evaluate(X[te], y[te])
        out_dir = args.out if tag == 'pgdnn' else args.out + '_' + tag
        model.save(out_dir, extra={'test': test, 'train_seconds': train_time, **summary})
        if out_dir != args.out:
            np.savez(os.path.join(out_dir, 'split_indices.npz'), train=tr, val=va, test=te)
        summary[tag] = {'lambdas': lam, 'test': test, 'train_seconds': train_time}
        print(f"[{tag}] 测试集: " + ", ".join(f"{k}={v:.4g}" for k, v in test.items()))

    with open(os.path.join(args.out, 'training_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
