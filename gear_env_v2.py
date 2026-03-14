import numpy as np
import gymnasium as gym
from gymnasium import spaces
from system import TransmissionSystem
# 假设 gear_calculations 已经包含了您之前修复的 clip 逻辑
from gear_calculations import GearCalculations 
from contact_stress import calculate_hertz_contact_stress
# 延迟导入PINN，避免模块级别导入PyTorch
import warnings
import re

# 忽略数学计算中的 RuntimeWarning，由环境内部逻辑处理
warnings.filterwarnings("ignore")

# 全局PINN模型实例（延迟加载）
_pinn_predictor = None

def get_pinn_predictor():
    """获取或加载PINN预测器（延迟导入PyTorch）"""
    global _pinn_predictor
    if _pinn_predictor is None:
        from pinn_contact_stress import PIINStressPredictor
        _pinn_predictor = PIINStressPredictor()
        _pinn_predictor.load()
    return _pinn_predictor

def predict_contact_stress_pinn(params):
    """
    使用PINN模型预测接触应力
    
    Args:
        params: 字典，包含PINN所需的12个输入参数（角度用度数）
            gama1, gama2, beta1, beta2, N1, N2, mn, Fn, kesai, an, E, nu
    
    Returns:
        dict: {'Pmax': float, 'a_len': float, 'valid': bool}
    """
    try:
        pinn = get_pinn_predictor()
        
        # 构造输入特征数组（注意顺序与训练时一致，角度用度数）
        X = np.array([[
            params['gama1'],      # 度
            params['gama2'],      # 度
            params['beta1'],      # 度
            params['beta2'],      # 度
            params['N1'],
            params['N2'],
            params['mn'],
            params['Fn'],
            params['kesai'],      # 度
            params['an'],         # 度
            params['E'],
            params['nu']
        ]], dtype=np.float32)
        
        pred = pinn.predict(X)
        return {
            'Pmax': float(pred[0, 0]),
            'a_len': float(pred[0, 1]),
            'valid': True
        }
    except Exception as e:
        print(f"PINN预测错误: {e}")  # 调试信息
        return {'Pmax': np.nan, 'a_len': np.nan, 'valid': False}

class GearEnvV2(gym.Env):
    """
    船用齿轮箱优化环境 V2 (稳健版)
    特点：
    1. 锚点重置机制 (Anchor Reset)：防止初始化参数不可计算。
    2. 分层课程奖励 (Hierarchical Curriculum)：生存 -> 合规 -> 性能。
    3. 动态锚点更新：跟随发现的优质参数区域移动出生点。
    """
    def __init__(self, base_params, curriculum_stage=0):
        super().__init__()
        self.base_params = base_params
        self.system = TransmissionSystem()
        
        # 定义智能体 ID
        self.agents = ['agent_gear1', 'agent_gear2', 'agent_gear3']
        self.num_agents = len(self.agents)
        
        # 动作空间：所有动作归一化为 [-1, 1]
        # 含义：在当前参数基础上进行微调，或者在参数范围内定位
        self.action_space = spaces.Dict({
            'agent_gear1': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            'agent_gear2': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            'agent_gear3': spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        })
        
        # 观测空间：[25] 维向量 (参数 + 指标 + 状态信息)
        self.obs_dim = 25
        self.observation_space = spaces.Dict({
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32) 
            for agent in self.agents
        })
        
        # 物理参数范围 (Min, Max)
        self.param_bounds = {
            'gama1': (4.5, 6.0),
            'gama2_gear1': (0.0, 1.5),
            'gama2_gear2': (1.5, 3.0),
            'beta3': (4.0, 20.0),
            'xt3': (-0.5, 0.5)
        }
        
        # 定义一个绝对安全的“锚点” (Safe Anchor)
        # 这是基于经验已知的可行解，用于初始化
        self.safe_anchor = {
            'gama1': 4.713305,       # 示例值
            'gama2_gear1': 0.277622,   # 示例值
            'gama2_gear2': 2.792749, # 示例值
            'beta3': 4.000539,         # 示例值
            'xt3': 0.117269            # 示例值
        }
        
        # 当前参数状态
        self.current_params = self.safe_anchor.copy()
        
        # 课程阶段: 0=生存, 1=合规, 2=性能
        self.stage = curriculum_stage
        
        self.steps = 0
        self.max_steps = 100
        self.last_info = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        
        # === 核心逻辑：基于锚点的受控重置 ===
        # 策略：绝大多数时候从已知安全点附近开始，极少数时候尝试大范围探索
        
        valid_reset = False
        attempt = 0
        
        while not valid_reset and attempt < 5:
            attempt += 1
            # 90% 的概率：保守模式，仅在锚点周围施加微小扰动 (0.5%)
            # 10% 的概率：探索模式，施加稍大扰动 (2%)
            noise_scale = 0.005 if np.random.random() < 0.9 else 0.02
            
            for k, anchor_val in self.safe_anchor.items():
                low, high = self.param_bounds[k]
                span = high - low
                noise = np.random.uniform(-noise_scale, noise_scale) * span
                self.current_params[k] = np.clip(anchor_val + noise, low, high)
            
            # 立即检查参数有效性
            if self._check_validity_internal():
                valid_reset = True
        
        # 如果尝试多次都失败，强制使用纯净锚点
        if not valid_reset:
            self.current_params = self.safe_anchor.copy()
            
        return self._get_obs(), {}

    def _check_validity_internal(self):
        """内部快速检查当前参数是否能计算"""
        metrics = self._calculate_metrics()
        return metrics.get('valid', False)

    def _denormalize(self, action_val, param_name):
        """将动作 [-1, 1] 映射回物理参数范围"""
        low, high = self.param_bounds[param_name]
        # 映射公式: low + (act + 1)/2 * (high - low)
        return low + (action_val + 1.0) * 0.5 * (high - low)

    def step(self, action_dict):
        self.steps += 1
        
        # 1. 解析动作并更新参数
        # 引入动量更新 (Momentum Update)，防止参数剧烈突变导致计算崩溃
        # 新参数 = 0.7 * 旧参数 + 0.3 * 动作目标
        alpha = 0.3 
        
        target_gama2_g1 = self._denormalize(action_dict['agent_gear1'][0], 'gama2_gear1')
        self.current_params['gama2_gear1'] = (1-alpha) * self.current_params['gama2_gear1'] + alpha * target_gama2_g1
        
        target_gama2_g2 = self._denormalize(action_dict['agent_gear2'][0], 'gama2_gear2')
        self.current_params['gama2_gear2'] = (1-alpha) * self.current_params['gama2_gear2'] + alpha * target_gama2_g2
        
        act_g3 = action_dict['agent_gear3']
        target_gama1 = self._denormalize(act_g3[0], 'gama1')
        target_beta3 = self._denormalize(act_g3[1], 'beta3')
        target_xt3 = self._denormalize(act_g3[2], 'xt3')
        
        self.current_params['gama1'] = (1-alpha) * self.current_params['gama1'] + alpha * target_gama1
        self.current_params['beta3'] = (1-alpha) * self.current_params['beta3'] + alpha * target_beta3
        self.current_params['xt3'] = (1-alpha) * self.current_params['xt3'] + alpha * target_xt3

        # 2. 计算系统指标
        metrics = self._calculate_metrics()
        
        # 3. 计算奖励
        reward = self._compute_reward(metrics)
        
        # 4. 状态判断
        terminated = False
        truncated = self.steps >= self.max_steps
        
        # === 核心逻辑：立即熔断与动态锚点 ===
        
        if not metrics.get('valid', False):
            # 如果进入无效区域，立即终止 Episode
            # 给一个较大的负惩罚，告诉 Agent "此路不通"
            reward = -50.0 
            terminated = True
        else:
            # 如果是有效区域，且奖励较高（说明发现了好参数），则更新锚点
            # 阈值根据实际情况调整，例如 > 5.0 (假设基础分是1.0)
            if reward > 5.0:
                # 缓慢移动锚点：Anchor = 0.995 * Anchor + 0.005 * Good_Params
                for k in self.safe_anchor:
                    self.safe_anchor[k] = 0.995 * self.safe_anchor[k] + 0.005 * self.current_params[k]
        
        # 5. 构建返回值
        obs = self._get_obs()
        
        self.last_info = {
            'metrics': metrics,
            'params': self.current_params.copy(),
            'curriculum_stage': self.stage
        }
        
        return obs, {a: reward for a in self.agents}, {a: terminated for a in self.agents}, {a: truncated for a in self.agents}, self.last_info

    def _calculate_metrics(self):
        """调用 System 计算并解析结果"""
        try:
            # 构造输入字典
            sys_input = {
                "gear1": {"N2": self.base_params["gear1"]["N2"], "gama2": self.current_params['gama2_gear1'], "b2": self.base_params["gear1"]["b2"]},
                "gear2": {"N2": self.base_params["gear2"]["N2"], "gama2": self.current_params['gama2_gear2'], "b2": self.base_params["gear2"]["b2"]},
                "gear3": {
                    "mn": self.base_params["gear3"]["mn"], "N1": self.base_params["gear3"]["N1"],
                    "gama1": self.current_params['gama1'], "beta1": self.current_params['beta3'], "xt1": self.current_params['xt3'],
                    "b1": self.base_params["gear3"]["b1"], "Ez": self.base_params["gear3"]["Ez"],
                    "han": self.base_params["gear3"]["han"], "cn": self.base_params["gear3"]["cn"], "an": self.base_params["gear3"]["an"],
                    "deta_13": self.base_params["gear3"]["deta_13"], "deta_23": self.base_params["gear3"]["deta_23"],
                    "offset_direction": self.base_params["gear3"]["offset_direction"],
                    "ajc_target": self.base_params["gear3"].get("ajc_target", 320.0),
                }
            }
            
            # 执行计算
            res = self.system.calculate_system(sys_input)
            
            # 检查返回值类型
            p1_result = res.get('gear_pair_1', {})
            p2_result = res.get('gear_pair_2', {})
            
            # 如果返回的是字符串（错误信息），转换为空字典
            if isinstance(p1_result, str):
                p1_result = {'text': p1_result}
            if isinstance(p2_result, str):
                p2_result = {'text': p2_result}
            
            # 解析文本结果
            p1_txt = p1_result.get('text', '')
            p2_txt = p2_result.get('text', '')
            
            # 检查报错关键词
            if "ERROR" in p1_txt or "ERROR" in p2_txt or "计算错误" in p1_txt or "计算错误" in p2_txt:
                return {'valid': False}
            
            # 提取数值
            vals = self._extract_values_from_text(p1_txt, p2_txt)
            
            # 检查 NaN（只检查原有的安全系数值）
            if any(np.isnan(v) for v in vals.values()):
                return {'valid': False}
            
            # 提取中心距值（允许为 NaN，不影响有效性判断）
            axj = p1_result.get('axj', np.nan)  # 相交轴中心距
            ajc = p2_result.get('ajc', np.nan)  # 交错轴中心距
            
            # 计算中心距误差
            axj_target = self.base_params["gear3"].get("axj_target", 344.0)
            ajc_target = self.base_params["gear3"].get("ajc_target", 320.0)
            
            vals['axj'] = axj if not np.isnan(axj) else 0.0
            vals['ajc'] = ajc if not np.isnan(ajc) else 0.0
            vals['axj_error'] = abs(axj - axj_target) if not np.isnan(axj) else 999.0
            vals['ajc_error'] = abs(ajc - ajc_target) if not np.isnan(ajc) else 999.0
            
            # ============ 接触应力计算（使用PINN预测）============
            try:
                # 获取FPD角（从已解析的结果中获取）
                kesai_R = vals.get('kesai_R', 0)  # 右齿面FPD角（度）
                kesai_L = vals.get('kesai_L', 0)  # 左齿面FPD角（度）
                
                # 公共材料参数
                E = self.base_params["gear3"].get("E", 210000)
                nu = self.base_params["gear3"].get("nu", 0.3)
                P = self.base_params["gear3"].get("P", 450)  # 功率 kW
                n1 = self.base_params["gear3"].get("n1", 2500)  # 转速 rpm
                mn = self.base_params["gear3"]["mn"]
                an = self.base_params["gear3"]["an"]  # 压力角（度）
                
                # ========== 齿轮对1：大齿轮3 + 小齿轮2（相交轴）==========
                N1_pair1 = self.base_params["gear3"]["N1"]
                N2_pair1 = self.base_params["gear2"]["N2"]
                gama1_pair1 = self.current_params['gama1']  # 度
                gama2_pair1 = self.current_params['gama2_gear2']  # 度
                beta1_pair1 = self.current_params['beta3']  # 度
                beta2_pair1 = -self.current_params['beta3']  # 度（反向）
                
                # 选择FPD角（根据接触面）
                contact_face_pair1 = self.base_params["gear2"].get("contact_face", "right")
                kesai_pair1 = kesai_R if contact_face_pair1 == "right" else kesai_L
                
                # 计算法向力Fn（使用小齿轮2计算）
                # Fn = 9550 * P / (n2 * r2 * cos(beta2) * cos(an))
                r2_pair1 = (mn * N2_pair1) / (2 * np.cos(np.radians(beta2_pair1)))
                n2_pair1 = n1  # 小齿轮转速
                Fn_pair1 = 9549 * P * 1000 / (n2_pair1 * r2_pair1 * np.cos(np.radians(beta2_pair1)) * np.cos(np.radians(an)))
                
                # PINN预测输入（12个参数，角度用度数，不考虑变位系数）
                pinn_params_pair1 = {
                    'gama1': gama2_pair1,'gama2': gama1_pair1,
                    'beta1': beta2_pair1,'beta2': beta1_pair1, 
                    'N1': N2_pair1,'N2': N1_pair1, 'mn': mn,
                    'Fn': abs(Fn_pair1), 'kesai': kesai_pair1, 'an': an,
                    'E': E, 'nu': nu
                }
                pinn_result_pair1 = predict_contact_stress_pinn(pinn_params_pair1)
                vals['Pmax_pair1'] = pinn_result_pair1['Pmax'] if pinn_result_pair1['valid'] else np.nan
                vals['a_len_pair1'] = pinn_result_pair1['a_len'] if pinn_result_pair1['valid'] else np.nan
                
                # ========== 齿轮对2：大齿轮3 + 小齿轮1（交错轴）==========
                N1_pair2 = self.base_params["gear3"]["N1"]
                N2_pair2 = self.base_params["gear1"]["N2"]
                gama1_pair2 = self.current_params['gama1']  # 度
                gama2_pair2 = self.current_params['gama2_gear1']  # 度
                beta1_pair2 = self.current_params['beta3']  # 度
                # 从计算结果获取小齿轮螺旋角（弧度制 → 度）
                beta2_rad = p2_result.get('beta2', -np.radians(beta1_pair2))  # 默认值为反向
                beta2_pair2 = np.degrees(beta2_rad)  # 转换为度
                
                # 选择FPD角（根据接触面）
                contact_face_pair2 = self.base_params["gear1"].get("contact_face", "left")
                kesai_pair2 = kesai_R if contact_face_pair2 == "right" else kesai_L
                
                # 计算法向力Fn（使用小齿轮1计算）
                r2_pair2 = (mn * N2_pair2) / (2 * np.cos(np.radians(beta2_pair2)))
                n2_pair2 = n1  # 小齿轮转速
                Fn_pair2 = 9550 * P * 1000 / (n2_pair2 * r2_pair2 * np.cos(np.radians(beta2_pair2)) * np.cos(np.radians(an)))
                
                # PINN预测输入
                pinn_params_pair2 = {
                    'gama1': gama2_pair2,'gama2': gama1_pair2,
                    'beta1': beta2_pair2,'beta2': beta1_pair2, 
                    'N1': N2_pair2,'N2': N1_pair2, 'mn': mn,
                    'Fn': abs(Fn_pair2), 'kesai': kesai_pair2, 'an': an,
                    'E': E, 'nu': nu
                }
                pinn_result_pair2 = predict_contact_stress_pinn(pinn_params_pair2)
                vals['Pmax_pair2'] = pinn_result_pair2['Pmax'] if pinn_result_pair2['valid'] else np.nan
                vals['a_len_pair2'] = pinn_result_pair2['a_len'] if pinn_result_pair2['valid'] else np.nan
                
                # Pmax取两对中的最大值（用于约束检查）
                Pmax1 = vals['Pmax_pair1'] if not np.isnan(vals['Pmax_pair1']) else 0
                Pmax2 = vals['Pmax_pair2'] if not np.isnan(vals['Pmax_pair2']) else 0
                vals['Pmax'] = max(Pmax1, Pmax2) if (Pmax1 > 0 or Pmax2 > 0) else np.nan
                
            except Exception as e:
                vals['Pmax'] = np.nan
                vals['Pmax_pair1'] = np.nan
                vals['Pmax_pair2'] = np.nan
                vals['a_len_pair1'] = np.nan
                vals['a_len_pair2'] = np.nan
            # ============ 接触应力计算结束 ============
            
            vals['valid'] = True
            return vals
            
        except Exception as e:
            # 捕获任何计算异常
            return {'valid': False}

    def _extract_values_from_text(self, t1, t2):
        """从 MATLAB/Backend 返回的文本中提取关键指标"""
        def get_val(txt, key):
            # 匹配 "key = value" 或 "key=value"
            match = re.search(rf"{key}.*?=\s*([-+]?\d*\.?\d+)", txt, re.IGNORECASE)
            return float(match.group(1)) if match else np.nan

        vals = {
            'kesai_R': get_val(t1, 'kesai_R'), # 角度（度）
            'kesai_L': get_val(t2, 'kesai_L'),
            # 提取所有安全系数
            'Sg1_1': get_val(t1, 'Sg1'), 'Sj1_1': get_val(t1, 'Sj1'),
            'Sg2_1': get_val(t1, 'Sg2'), 'Sj2_1': get_val(t1, 'Sj2'),
            'Sg2_2': get_val(t2, 'Sg2'), 'Sj2_2': get_val(t2, 'Sj2')
        }
        
        # 计算最小安全系数
        vals['min_safety'] = min(
            vals['Sg1_1'], vals['Sj1_1'], vals['Sg2_1'], vals['Sj2_1'], 
            vals['Sg2_2'], vals['Sj2_2']
        )
        return vals

    def _compute_reward(self, metrics):
        """分层课程奖励计算"""
        if not metrics.get('valid', False):
            return -10.0 # 基础惩罚
        
        # 1. 基础生存分 (Stage 0)
        r_exist = 2.0 
        
        # 2. 合规分 (Stage 1) - 关注安全系数
        # 目标: min_safety >= 1.0
        min_safe = metrics.get('min_safety', 0)
        # 使用 Sigmoid 风格映射: <1.0 时惩罚，>1.0 时奖励
        # 范围大致在 [-1, 1] 之间
        r_constraint = np.tanh((min_safe - 1.0) * 2.0)
        
        # 3. 性能分 (Stage 2) - 关注接触印痕角度
        # 目标: 角度越接近 0 越好
        kR = abs(metrics.get('kesai_R', 20.0))
        kL = abs(metrics.get('kesai_L', 20.0))
        # 20度 -> 0分, 0度 -> 2分
        r_perf = (2.0 - 0.1 * kR) + (2.0 - 0.1 * kL)
        
        # 课程逻辑
        if self.stage == 0:
            # 只要能算出来，就给正分
            return r_exist
        elif self.stage == 1:
            # 生存 + 安全系数
            return r_exist + 2.0 * r_constraint
        else:
            # 全面优化
            # 只有当安全系数 > 1.0 时，性能分才生效，否则主要优化安全系数
            if min_safe < 1.0:
                 return r_exist + 3.0 * r_constraint # 强力惩罚不安全
            else:
                 return r_exist + r_constraint + r_perf

    def _get_obs(self):
        """构建归一化观测向量"""
        obs = []
        # 1. 当前参数 (归一化到 [-1, 1])
        for k in ['gama1', 'gama2_gear1', 'gama2_gear2', 'beta3', 'xt3']:
            low, high = self.param_bounds[k]
            val = self.current_params.get(k, (low+high)/2)
            norm_val = 2 * (val - low) / (high - low) - 1.0
            obs.append(norm_val)
            
        # 2. 补充信息 (例如阶段信息，帮助 Critic 判断)
        obs.append(self.stage / 2.0) # 归一化阶段
        
        # 3. 填充到 25 维
        obs = np.array(obs + [0.0] * (25 - len(obs)), dtype=np.float32)
        
        return {a: obs for a in self.agents}

    def set_curriculum_stage(self, stage):
        self.stage = stage
        print(f"\n[Environment] Switched to Curriculum Stage: {stage}")