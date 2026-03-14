import numpy as np
from gear_calculations import GearCalculations

class TransmissionSystem:
    """传动系统计算类 - 大齿轮3分别与两个小齿轮啮合"""
    
    def __init__(self):
        self.gear_calculator = GearCalculations()
        self.results = {}
    
    def calculate_system(self, system_params):
        """
        计算整个传动系统
        
        Args:
            system_params: 系统参数字典，包含所有齿轮对的参数
            
        Returns:
            dict: 包含所有计算结果的字典
        """
        try:
            # 提取系统参数
            gear1_params = system_params["gear1"]  # 小齿轮1参数（交错轴变厚齿轮）
            gear2_params = system_params["gear2"]  # 小齿轮2参数（相交轴变厚齿轮）
            gear3_params = system_params["gear3"]  # 大齿轮3参数
            
            # 计算齿轮对1：大齿轮3与小齿轮2（相交轴变厚齿轮）
            pair1_result = self._calculate_intersecting_axis_pair(gear3_params, gear2_params)   
            
            # 计算齿轮对2：大齿轮3与小齿轮1（交错轴变齿厚与圆柱齿轮）
            pair2_result = self._calculate_crossed_axis_pair(gear3_params, gear1_params)
            
            # 整理结果
            self.results = {
                "gear_pair_1": pair1_result,
                "gear_pair_2": pair2_result,
            }
            
            return self.results
            
        except Exception as e:
            return f"系统计算错误: {str(e)}"
    
    def _calculate_intersecting_axis_pair(self, gear3_params, gear2_params):
        """计算相交轴变厚齿轮对（大齿轮3与小齿轮2）"""
        # 构建相交轴变厚齿轮计算参数
        intersecting_params = {
            "N1": gear3_params["N1"],
            "N2": gear2_params["N2"],
            "mn": gear3_params["mn"],
            "deta": gear3_params["deta_23"],  # 大齿轮3与小齿轮2的轴交角
            "gama1": gear3_params["gama1"],
            "gama2": gear2_params["gama2"],
            "han": gear3_params["han"],
            "cn": gear3_params["cn"],
            "xt1": gear3_params["xt1"],
            "b1": gear3_params["b1"],
            "b2": gear2_params["b2"],
            "beta1": gear3_params["beta1"],
            "an": gear3_params["an"]
        }
        
        # 调用相交轴变厚齿轮计算方法
        result = self.gear_calculator.intersecting_axis_calculation(intersecting_params)
        
        
        return result
    
    def _calculate_crossed_axis_pair(self, gear3_params, gear1_params):
        """计算交错轴变齿厚齿轮副（大齿轮3与小齿轮1）"""
        # 构建交错轴变齿厚齿轮副计算参数
        crossed_params = {
            "N1": gear3_params["N1"],
            "N2": gear1_params["N2"],
            "mn": gear3_params["mn"],
            "deta": gear3_params["deta_13"],  # 大齿轮3与小齿轮1的轴交角
            "gama1": gear3_params["gama1"],
            "gama2": gear1_params["gama2"],
            "han": gear3_params["han"],
            "cn": gear3_params["cn"],
            "xt1": gear3_params["xt1"],
            "b1": gear3_params["b1"],
            "b2": gear1_params["b2"],
            "beta1": gear3_params["beta1"],
            "an": gear3_params["an"],
            "Ez": gear3_params["Ez"],
            "offset_direction":gear3_params["offset_direction"],
            "ajc_target": gear3_params.get("ajc_target", 320.0),  # 传递交错轴目标中心距
        }
        
        # 调用交错轴变齿厚与圆柱齿轮计算方法
        result = self.gear_calculator.crossed_axis_variable_thickness_calculation(crossed_params)
        

        
        return result

# 使用示例
def create_example_system():
    """创建示例传动系统参数"""
    system_params = {
        "gear1": {  # 小齿轮1参数（交错轴变厚齿轮）- 齿轮对2
            "N2": 24,           # 齿数
            "gama2": 1.25,        # 节锥角
            "b2": 130,           # 齿宽
            "contact_face": "left",  # 接触应力计算齿面: "left" 或 "right"
        },
        "gear2": {  # 小齿轮2参数（相交轴变厚齿轮）- 齿轮对1
            "N2": 24,           # 齿数
            "gama2":2,        # 节锥角
            "b2": 130,           # 齿宽
            "contact_face": "right",  # 接触应力计算齿面: "left" 或 "right"
        },
        "gear3": {  # 大齿轮3参数
            "mn": 6,          # 法向模数
            "N1": 61,           # 齿数
            "gama1": 5,       # 节锥角
            "beta1": 15,      # 螺旋角，一个示例值
            "xt1":0,         # 变位系数，一个示例值
            "b1": 125,           # 齿宽
            "Ez": 180,       # 主偏置距

            "han": 1.0,         # 齿顶高系数
            "cn": 0.25,         # 顶隙系数  
            "an": 20,            # 压力角
            "deta_13":7,      # 与小齿轮1的轴交角
            "deta_23":7,      # 与小齿轮2的轴交角
            "offset_direction":"right",
            "axj_target": 316.27,  # 相交轴目标中心距 (mm)
            "ajc_target": 320.42,  # 交错轴目标中心距 (mm)
            
            # 材料和工况参数（用于接触应力计算）
            "E": 210000,        # 杨氏模量 (MPa)
            "nu": 0.3,          # 泊松比
            "P": 850,           # 功率 (kW)
            "n1": 2500,         # 小齿轮转速 (rpm)
            "Pmax_limit": 1500, # 许用接触应力上限 (MPa)
        }    
    }
    return system_params

if __name__ == "__main__":
    import re
    
    # 创建传动系统实例
    system = TransmissionSystem()
    # 创建示例参数
    params = create_example_system()
    # 计算系统
    results = system.calculate_system(params)
    
    # 打印齿轮几何计算结果
    p1_result = results['gear_pair_1']
    p2_result = results['gear_pair_2']
    print("=" * 60)
    print("齿轮对1（相交轴）计算结果：")
    print(p1_result['text'])
    print("=" * 60)
    print("齿轮对2（交错轴）计算结果：")
    print(p2_result['text'])
    
    # ============ 使用PINN模型计算接触应力 ============
    print("=" * 60)
    print("使用PINN模型计算接触应力...")
    
    from pinn_numpy_inference import PINNPredictor
    
    # 加载模型（纯NumPy，无需PyTorch）
    pinn = PINNPredictor('pinn_model_numpy')
    print("PINN模型加载成功（纯NumPy推理）")
    
    # 提取公共参数
    gear3 = params["gear3"]
    E = gear3.get("E", 210000)
    nu = gear3.get("nu", 0.3)
    P = gear3.get("P", 500)       # 功率 kW
    n1 = gear3.get("n1", 2500)    # 转速 rpm
    mn = gear3["mn"]
    an = gear3["an"]               # 压力角（度）
    
    # 从文本中提取FPD角
    def get_val(txt, key):
        match = re.search(rf"{key}.*?=\s*([-+]?\d*\.?\d+)", txt, re.IGNORECASE)
        return float(match.group(1)) if match else 0.0
    
    p1_txt = p1_result.get('text', '')
    p2_txt = p2_result.get('text', '')
    kesai_R = get_val(p1_txt, 'kesai_R')
    kesai_L = get_val(p2_txt, 'kesai_L')
    
    print(f"\n  右齿面FPD角 kesai_R = {kesai_R:.4f}°")
    print(f"  左齿面FPD角 kesai_L = {kesai_L:.4f}°")
    
    # ========== 齿轮对1：大齿轮3 + 小齿轮2（相交轴）==========
    N1_p1 = gear3["N1"]
    N2_p1 = params["gear2"]["N2"]
    gama1_p1 = gear3["gama1"]       # 度
    gama2_p1 = params["gear2"]["gama2"]  # 度
    beta1_p1 = gear3["beta1"]       # 度
    beta2_p1 = -gear3["beta1"]      # 度（相交轴反向）
    
    # 选择FPD角
    contact_face_p1 = params["gear2"].get("contact_face", "right")
    kesai_p1 = kesai_R if contact_face_p1 == "right" else kesai_L
    
    # 计算法向力Fn
    r2_p1 = (mn * N2_p1) / (2 * np.cos(np.radians(beta2_p1)))
    Fn_p1 = 9549 * P * 1000 / (n1 * r2_p1 * np.cos(np.radians(beta2_p1)) * np.cos(np.radians(an)))
    
    # PINN预测（小齿轮视角输入）
    X_p1 = np.array([[
        gama2_p1, gama1_p1,         # gama1=小轮, gama2=大轮
        beta2_p1, beta1_p1,          # beta1=小轮, beta2=大轮
        N2_p1, N1_p1, mn,            # N1=小轮, N2=大轮
        abs(Fn_p1), kesai_p1, an,
        E, nu
    ]], dtype=np.float32)
    pred_p1 = pinn.predict(X_p1)
    
    print(f"\n--- 齿轮对1（相交轴：大齿轮3 + 小齿轮2）---")
    print(f"  接触面: {contact_face_p1}, FPD角: {kesai_p1:.4f}°")
    print(f"  法向力 Fn = {abs(Fn_p1):.2f} N")
    print(f"  最大接触应力 Pmax = {pred_p1[0,0]:.2f} MPa")
    print(f"  接触椭圆长轴 a_len = {pred_p1[0,1]:.2f} mm")
    
    # ========== 齿轮对2：大齿轮3 + 小齿轮1（交错轴）==========
    N1_p2 = gear3["N1"]
    N2_p2 = params["gear1"]["N2"]
    gama1_p2 = gear3["gama1"]       # 度
    gama2_p2 = params["gear1"]["gama2"]  # 度
    beta1_p2 = gear3["beta1"]       # 度
    # 从计算结果获取小齿轮螺旋角（弧度制 → 度）
    beta2_rad = p2_result.get('beta2', -np.radians(beta1_p2))
    beta2_p2 = np.degrees(beta2_rad)
    
    # 选择FPD角
    contact_face_p2 = params["gear1"].get("contact_face", "left")
    kesai_p2 = kesai_R if contact_face_p2 == "right" else kesai_L
    
    # 计算法向力Fn
    r2_p2 = (mn * N2_p2) / (2 * np.cos(np.radians(beta2_p2)))
    Fn_p2 = 9550 * P * 1000 / (n1 * r2_p2 * np.cos(np.radians(beta2_p2)) * np.cos(np.radians(an)))
    
    # PINN预测（小齿轮视角输入）
    X_p2 = np.array([[
        gama2_p2, gama1_p2,
        beta2_p2, beta1_p2,
        N2_p2, N1_p2, mn,
        abs(Fn_p2), kesai_p2, an,
        E, nu
    ]], dtype=np.float32)
    pred_p2 = pinn.predict(X_p2)
    
    print(f"\n--- 齿轮对2（交错轴：大齿轮3 + 小齿轮1）---")
    print(f"  接触面: {contact_face_p2}, FPD角: {kesai_p2:.4f}°")
    print(f"  小齿轮螺旋角 beta2 = {beta2_p2:.4f}°")
    print(f"  法向力 Fn = {abs(Fn_p2):.2f} N")
    print(f"  最大接触应力 Pmax = {pred_p2[0,0]:.2f} MPa")
    print(f"  接触椭圆长轴 a_len = {pred_p2[0,1]:.2f} mm")
    
    # 汇总
    Pmax_max = max(pred_p1[0,0], pred_p2[0,0])
    Pmax_limit = gear3.get("Pmax_limit", 1500)
    print(f"\n{'=' * 60}")
    print(f"  最大接触应力 Pmax = {Pmax_max:.2f} MPa (许用值: {Pmax_limit} MPa)")
    if Pmax_max < Pmax_limit:
        print(f"  ✅ 接触应力满足要求")
    else:
        print(f"  ⚠️ 接触应力超过许用值！")
    print(f"{'=' * 60}")
