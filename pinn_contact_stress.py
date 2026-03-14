"""
物理信息神经网络(PINN) - 接触应力预测模型
使用物理约束增强的神经网络预测齿轮接触应力Pmax
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import joblib

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContactStressPINN(nn.Module):
    """
    物理信息神经网络 - 接触应力预测
    
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
    PINN接触应力预测器 - 管理模型训练和推理
    """
    
    # 输入特征顺序 (12维) - 移除xt1/xt2，计算时默认为0
    FEATURE_NAMES = [
        'gama1', 'gama2', 'beta1', 'beta2', 'N1', 'N2', 'mn',
        'Fn', 'kesai', 'an', 'E', 'nu'
    ]
    
    # 输出特征顺序 (2维)
    OUTPUT_NAMES = ['Pmax', 'a_len']
    
    def __init__(self, learning_rate=1e-3):
        self.model = ContactStressPINN(input_dim=12, output_dim=2).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # 数据标准化参数
        self.scaler_mean = None
        self.scaler_scale = None
        self.output_mean = None
        self.output_scale = None
        
        # 物理约束权重
        self.physics_weight = 0.01
    
    def fit_scaler(self, X, y):
        """计算标准化参数"""
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
    
    def physics_loss(self, X, y_pred):
        """
        物理约束损失
        
        约束:
        1. Pmax > 0, a_len > 0 (正值约束)
        2.1 ∂Pmax/∂Fn > 0 (法向力越大，应力越大)
        2.2 ∂Pmax/∂|kesai| > 0 (FPD角绝对值越大，应力越大)
        3. a_len随Fn增大而增大 (∂a_len/∂Fn > 0)
        """
        loss = torch.tensor(0.0, device=device)
        
        # y_pred[:, 0] = Pmax, y_pred[:, 1] = a_len
        Pmax_pred = y_pred[:, 0:1]
        a_len_pred = y_pred[:, 1:2]
        
        y_real = self.inverse_transform_output(y_pred)
        Pmax_real = y_real[:, 0:1]
        a_len_real = y_real[:, 1:2]
        
        # 1. 正值约束
        loss += torch.mean(torch.relu(-Pmax_real))
        loss += torch.mean(torch.relu(-a_len_real))  
        # 2. 梯度约束
        if X.requires_grad:
            grad_outputs = torch.ones_like(Pmax_pred)
            grads = torch.autograd.grad(
                outputs=Pmax_pred, inputs=X,
                grad_outputs=grad_outputs,
                create_graph=True, retain_graph=True
            )[0]
            
            # ∂Pmax/∂Fn > 0 (Fn索引=9)
            Fn_grad = grads[:, 9]
            loss += 0.01 * torch.mean(torch.relu(-Fn_grad))
            
            # ∂Pmax/∂|kesai| > 0 (kesai索引=10)
            kesai_grad = grads[:, 10]
            kesai_val = X[:, 10]
            loss += 0.01 * torch.mean(torch.relu(-kesai_grad * kesai_val))
        
        # 3. a_len随Fn增大而增大 (∂a_len/∂Fn > 0)
        if X.requires_grad:
            a_len_grad_outputs = torch.ones_like(a_len_pred)
            a_len_grads = torch.autograd.grad(
                outputs=a_len_pred, inputs=X,
                grad_outputs=a_len_grad_outputs,
                create_graph=True, retain_graph=True
            )[0]
            # Fn索引=9
            a_len_Fn_grad = a_len_grads[:, 9]
            loss += 0.01 * torch.mean(torch.relu(-a_len_Fn_grad))
        
        return loss
    
    def train(self, X_train, y_train, epochs=500, batch_size=32, verbose=True):
        """
        训练PINN模型
        
        Args:
            X_train: numpy array, shape (N, 14)
            y_train: numpy array, shape (N,) - Pmax值
            epochs: 训练轮数
            batch_size: 批大小
        """
        # 标准化
        self.fit_scaler(X_train, y_train)
        X_scaled = self.transform_input(X_train).astype(np.float32)
        y_scaled = self.transform_output(y_train).astype(np.float32)
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        
        # 创建数据加载器
        dataset = TensorDataset(
            torch.tensor(X_scaled),
            torch.tensor(y_scaled)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 损失函数
        mse_loss = nn.MSELoss()
        
        self.model.train()
        loss_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_physics_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                batch_X.requires_grad = True
                
                self.optimizer.zero_grad()
                
                # 前向传播
                y_pred = self.model(batch_X)
                
                # 加权MSE损失 - a_len权重更高以平衡量级差异
                # Pmax~1000 MPa, a_len~50 mm，量级差20倍
                pmax_loss = mse_loss(y_pred[:, 0], batch_y[:, 0])
                alen_loss = mse_loss(y_pred[:, 1], batch_y[:, 1])
                data_loss = pmax_loss + 2.5 * alen_loss
                
                # 物理损失
                phys_loss = self.physics_loss(batch_X, y_pred)
                
                # 总损失
                total_loss = data_loss + self.physics_weight * phys_loss
                
                # 反向传播
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += data_loss.item()
                epoch_physics_loss += phys_loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            avg_phys = epoch_physics_loss / len(dataloader)
            loss_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}, Physics: {avg_phys:.6f}")
        
        return loss_history
    
    def predict(self, X):
        """
        预测接触应力
        
        Args:
            X: numpy array, shape (N, 14) 或 dict
            
        Returns:
            Pmax预测值, shape (N,)
        """
        if isinstance(X, dict):
            # 从字典构建输入
            X = self._dict_to_array(X)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_scaled = self.transform_input(X).astype(np.float32)
        X_tensor = torch.tensor(X_scaled).to(device)
        
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor)
        
        y_real = self.inverse_transform_output(y_pred.cpu().numpy())
        return y_real
    
    def _dict_to_array(self, params):
        """将参数字典转换为数组"""
        arr = np.zeros(14)
        for i, name in enumerate(self.FEATURE_NAMES):
            arr[i] = params.get(name, 0)
        return arr
    
    def save(self, save_dir='pinn_model'):
        """保存模型"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'model.pth'))
        joblib.dump({
            'scaler_mean': self.scaler_mean,
            'scaler_scale': self.scaler_scale,
            'output_mean': self.output_mean,
            'output_scale': self.output_scale
        }, os.path.join(save_dir, 'scaler.pkl'))
        print(f"模型已保存至 {save_dir}")
    
    def load(self, save_dir='pinn_model'):
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


def generate_training_data(n_samples=2000, save_path='pinn_training_data.csv'):
    """
    生成PINN训练数据
    使用精确计算生成接触应力值
    """
    from contact_stress import calculate_hertz_contact_stress
    
    print(f"开始生成 {n_samples} 个训练样本...")
    
    # 参数范围 (12个输入特征，移除xt1/xt2)
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
    
    # 固定变位系数为0
    fixed_xt1 = 0.0
    fixed_xt2 = 0.0
    
    # 固定参数
    fixed_params = {
        'b1': 188,
        'b2': 180,
        'fuhao': -1,
    }
    
    # Latin Hypercube采样
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=len(param_ranges))
    samples = sampler.random(n=n_samples)
    
    data = []
    valid_count = 0
    
    for i, sample in enumerate(samples):
        params = fixed_params.copy()
        
        j = 0
        for name, (low, high) in param_ranges.items():
            if name in ['N1', 'N2']:
                params[name] = int(low + sample[j] * (high - low))
            else:
                params[name] = low + sample[j] * (high - low)
            j += 1
        
        # 保存原始角度值用于输出
        gama1_deg = params['gama1']
        gama2_deg = params['gama2']
        beta1_deg = params['beta1']
        beta2_deg = params['beta2']
        an_deg = params['an']
        
        # 转换角度为弧度
        params['gama1'] = np.radians(gama1_deg)
        params['gama2'] = np.radians(gama2_deg)
        params['beta1'] = np.radians(beta1_deg)
        params['beta2'] = np.radians(beta2_deg)
        params['an'] = np.radians(an_deg)
        params['sigma_rad'] = np.radians(params['kesai'])
        params['xt1'] = fixed_xt1  # 固定变位系数为0
        params['xt2'] = fixed_xt2
        params['drive_gear'] = 1
        
        # 计算接触应力
        result = calculate_hertz_contact_stress(params)
        
        if result.get('valid', False) and not np.isnan(result.get('Pmax', np.nan)):
            Pmax = result['Pmax']
            a_len = result.get('a_len', 0)
            if 0 < Pmax and 0<a_len:
                data.append({
                    'gama1': gama1_deg,
                    'gama2': gama2_deg,
                    'beta1': beta1_deg,
                    'beta2': beta2_deg,
                    'N1': params['N1'],
                    'N2': params['N2'],
                    'mn': params['mn'],
                    'Fn': params['Fn'],
                    'kesai': params['kesai'],
                    'an': an_deg,
                    'E': params['E'],
                    'nu': params['nu'],
                    'Pmax': Pmax,
                    'a_len': a_len
                })
                valid_count += 1
        
        if (i + 1) % 200 == 0:
            print(f"  进度: {i+1}/{n_samples}, 有效样本: {valid_count}")
    
    # 保存数据
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"数据生成完成: {valid_count} 个有效样本，保存至 {save_path}")
    
    return df


if __name__ == "__main__":
    # 1. 生成训练数据（如果不存在）
    data_path = 'pinn_training_data.csv'
    
    if not os.path.exists(data_path):
        print("未找到训练数据，开始生成...")
        df = generate_training_data(n_samples=3500, save_path=data_path)
    else:
        print(f"加载现有训练数据: {data_path}")
        df = pd.read_csv(data_path)
        # 转换为数值类型并清除NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        # 过滤到合理范围
        df = df[(df['Pmax'] >= 400) & (df['Pmax'] <= 2500)& (df['a_len'] <= 100)]
        print(f"过滤后: {len(df)} 个有效样本")
    
    print(f"训练数据: {len(df)} 个样本")
    print(f"Pmax范围: [{df['Pmax'].min():.1f}, {df['Pmax'].max():.1f}] MPa")
    print(f"a_len范围: [{df['a_len'].min():.1f}, {df['a_len'].max():.1f}] mm")
    
    # 2. 准备训练数据
    feature_cols = PIINStressPredictor.FEATURE_NAMES
    output_cols = PIINStressPredictor.OUTPUT_NAMES
    X = df[feature_cols].values.astype(np.float32)
    y = df[output_cols].values.astype(np.float32)
    
    # 3. 训练PINN
    print("\n开始训练PINN模型...")
    pinn = PIINStressPredictor()
    loss_history = pinn.train(X, y, epochs=2000, batch_size=128)
    
    # 4. 保存模型
    pinn.save()
    
    # 5. 测试预测
    print("\n测试预测:")
    test_idx = np.random.choice(len(df), 5)
    for idx in test_idx:
        X_test = X[idx:idx+1]
        y_true = y[idx]
        y_pred = pinn.predict(X_test)[0]
        Pmax_err = abs(y_pred[0] - y_true[0]) / y_true[0] * 100
        a_len_err = abs(y_pred[1] - y_true[1]) / y_true[1] * 100
        print(f"  Pmax: 真实={y_true[0]:.1f}, 预测={y_pred[0]:.1f} MPa, 误差={Pmax_err:.1f}%")
        print(f"  a_len: 真实={y_true[1]:.2f}, 预测={y_pred[1]:.2f} mm, 误差={a_len_err:.1f}%")
