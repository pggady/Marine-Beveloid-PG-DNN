# 先导入PyTorch，确保在其他库之前加载（避免DLL冲突）
import torch
import optuna
import numpy as np
import os
import sys
import csv
from datetime import datetime

# 引入环境定义
from gear_env_v2 import GearEnvV2
from system import create_example_system

def objective(trial):
    """
    Optuna 目标函数
    目标：最小化 (接触角之和)
    约束：最小安全系数 > 1.5
    """
    # 1. 定义搜索空间
    gama1 = trial.suggest_float('gama1', 4, 6)
    gama2_g1 = trial.suggest_float('gama2_gear1', 0, 3.5)
    gama2_g2 = trial.suggest_float('gama2_gear2', 0, 3.5)
    beta3 = trial.suggest_float('beta3', 5.0,25.0)
    xt3 = trial.suggest_float('xt3', -0.5, 0.5)

    # 2. 实例化环境并注入参数
    base_params = create_example_system()
    env = GearEnvV2(base_params)
    
    env.current_params['gama1'] = gama1
    env.current_params['gama2_gear1'] = gama2_g1
    env.current_params['gama2_gear2'] = gama2_g2
    env.current_params['beta3'] = beta3
    env.current_params['xt3'] = xt3

    # 3. 计算
    metrics = env._calculate_metrics()
    
    # 4. 记录额外的中间指标 (User Attributes)
    # 这样在 CSV 里不仅能看到 Loss，还能看到具体的安全系数、角度和接触应力
    trial.set_user_attr("valid", metrics.get('valid', False))
    trial.set_user_attr("min_safety", metrics.get('min_safety', 0.0))
    trial.set_user_attr("kesai_L", metrics.get('kesai_L', 0.0))
    trial.set_user_attr("kesai_R", metrics.get('kesai_R', 0.0))
    trial.set_user_attr("Pmax", metrics.get('Pmax', 0.0))
    trial.set_user_attr("Pmax_pair1", metrics.get('Pmax_pair1', 0.0))
    trial.set_user_attr("Pmax_pair2", metrics.get('Pmax_pair2', 0.0))
    trial.set_user_attr("a_len_pair1", metrics.get('a_len_pair1', 0.0))
    trial.set_user_attr("a_len_pair2", metrics.get('a_len_pair2', 0.0))

    # 5. 剪枝
    if not metrics.get('valid', False):
        raise optuna.exceptions.TrialPruned()
    
    # 6. 计算 Loss (多目标优化)
    min_safety = metrics.get('min_safety', 0.0)
    kesai_sum = abs(metrics.get('kesai_R', 20.0)) + abs(metrics.get('kesai_L', 20.0))
    Pmax_pair1 = metrics.get('Pmax_pair1', 2000.0)
    Pmax_pair2 = metrics.get('Pmax_pair2', 2000.0)
    target_safety = 1.25
    Pmax_limit = 1500.0  # 许用接触应力上限 MPa
    
    if min_safety < target_safety:
        # 安全系数惩罚项 (最高优先级)
        loss = 100.0 + (target_safety - min_safety) * 100.0
    elif np.isnan(Pmax_pair1) or np.isnan(Pmax_pair2):
        # 接触应力计算失败，给予中等惩罚
        loss = 50.0 + kesai_sum
    else:
        # 多目标优化：角度 + 两对齿轮的接触应力
        # 角度项：kesai_sum范围约[0, 40]，归一化到[0, 1]
        angle_term = kesai_sum / 40.0
        # 接触应力项：两对齿轮的平均应力，归一化到[0, 1]范围
        stress_term1 = min(Pmax_pair1 / Pmax_limit, 2.0)
        stress_term2 = min(Pmax_pair2 / Pmax_limit, 2.0)
        stress_term = (stress_term1 + stress_term2) / 2.0
        # 综合loss：角度权重0.6，应力权重0.4
        loss = 0.6 * angle_term + 0.4 * stress_term
        
    return loss

def save_results_to_csv(study, filename="optuna_results.csv"):
    """将 Optuna 的运行历史保存为 CSV (与 baseline 格式一致)"""
    print(f"\n正在导出数据到 {filename} ...")
    
    # 使用与 baseline 一致的格式，添加接触应力相关字段
    fieldnames = ['number', 'gama1', 'gama2_gear1', 'gama2_gear2', 'beta3', 'xt3',
                  'value', 'min_safety', 'kesai_L', 'kesai_R', 'Pmax', 'Pmax_pair1', 'Pmax_pair2',
                  'a_len_pair1', 'a_len_pair2', 'valid']
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for trial in study.trials:
            # 跳过被剪枝的试验
            if trial.state.name == 'PRUNED':
                valid = False
                value = 200.0
            else:
                valid = trial.user_attrs.get('valid', False)
                value = trial.value if trial.value is not None else 200.0
            
            row = {
                'number': trial.number + 1,  # 从 1 开始编号
                'gama1': trial.params.get('gama1', ''),
                'gama2_gear1': trial.params.get('gama2_gear1', ''),
                'gama2_gear2': trial.params.get('gama2_gear2', ''),
                'beta3': trial.params.get('beta3', ''),
                'xt3': trial.params.get('xt3', ''),
                'value': value,
                'min_safety': trial.user_attrs.get('min_safety', 0.0),
                'kesai_L': trial.user_attrs.get('kesai_L', ''),
                'kesai_R': trial.user_attrs.get('kesai_R', ''),
                'Pmax': trial.user_attrs.get('Pmax', ''),
                'Pmax_pair1': trial.user_attrs.get('Pmax_pair1', ''),
                'Pmax_pair2': trial.user_attrs.get('Pmax_pair2', ''),
                'a_len_pair1': trial.user_attrs.get('a_len_pair1', ''),
                'a_len_pair2': trial.user_attrs.get('a_len_pair2', ''),
                'valid': valid
            }
            writer.writerow(row)
    
    print(f"✅ 导出成功: {filename}")

def run_optimization():
    print("="*50)
    print("开始 Optuna 极速优化 (目标: Safety > 1.5, Min Angles)")
    print("="*50)
    # 1. 创建 Study
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=20) # 前20次是随机探索
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
    # === 关键步骤：注入您的“安全锚点”作为初始值 ===
    # 告诉 TPE：第一步先跑这组参数，不要随机！
    study.enqueue_trial({
        'gama1': 5.45,       
        'gama2_gear1': 0.0,   
        'gama2_gear2': 2.3, 
        'beta3': 4.0,         
        'xt3': 0.5            
    })
    sampler = optuna.samplers.TPESampler(seed=42) 
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 运行 500 次
    study.optimize(objective, n_trials=1000, show_progress_bar=True)
    
    # 记录结束时间
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # --- 核心：保存数据 ---
    save_results_to_csv(study, "optuna_results.csv")
    
    # 保存运行时间
    with open("tpe_runtime.txt", 'w') as f:
        f.write(f"{duration:.2f}")
    print(f"✅ 运行时间已保存: tpe_runtime.txt ({duration:.2f}秒)")
    
    print("\n" + "="*50)
    print("🎉 优化完成！")
    print(f"  总耗时: {duration:.2f} 秒")
    
    best = study.best_trial
    print(f"最佳 Loss: {best.value:.4f}")
    print("最佳参数:")
    for k, v in best.params.items():
        print(f"  {k}: {v:.6f}")
    kL = best.user_attrs['kesai_L']
    kR = best.user_attrs['kesai_R']
    Pmax_pair1 = best.user_attrs.get('Pmax_pair1', 0.0)
    Pmax_pair2 = best.user_attrs.get('Pmax_pair2', 0.0)
    a_len_pair1 = best.user_attrs.get('a_len_pair1', 0.0)
    a_len_pair2 = best.user_attrs.get('a_len_pair2', 0.0)
    print(f"  最小安全系数: {best.user_attrs['min_safety']:.4f}")
    print(f"  左齿面角度: {kL:.4f}°")
    print(f"  右齿面角度: {kR:.4f}°")
    print(f"  齿轮对1接触应力 Pmax_pair1: {Pmax_pair1:.2f} MPa")
    print(f"  齿轮对2接触应力 Pmax_pair2: {Pmax_pair2:.2f} MPa")
    print(f"  齿轮对1接触长度: {a_len_pair1:.4f} mm")
    print(f"  齿轮对2接触长度: {a_len_pair2:.4f} mm")
    print("\n📊 提示：")
    print("请使用 Excel 打开 'optuna_results.csv' 查看所有数据。")
    print("你可以选中 'value' (Loss) 列绘制折线图，观察收敛过程。")
    print("="*50)

if __name__ == "__main__":
    run_optimization()