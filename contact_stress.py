"""
接触应力计算模块 - 赫兹接触理论
纯Python实现 - 使用SymPy符号计算实现精确曲率
"""

import numpy as np
from scipy.optimize import fsolve, brentq
from scipy.special import ellipk, ellipe

# 接触椭圆短长轴比的下限（v9：由 0.01 降到 1e-4）
K_MIN = 1e-4
import sympy as sp
from functools import lru_cache

# SymPy符号定义（全局，避免重复创建）
_l_sym, _u_sym = sp.symbols('l u', real=True)


@lru_cache(maxsize=128)
def _get_curvature_function(gama_val, beta_val, an_val, mn_val, N_val, fuhao_val):
    """
    生成曲率计算函数（使用SymPy符号微分，缓存以提高效率）
    
    MATLAB qulvjisuan.m的精确Python实现
    """
    l, u = _l_sym, _u_sym
    
    # 转换为Python数值
    gama = float(gama_val)
    beta = float(beta_val)
    an = float(an_val)
    mn = float(mn_val)
    N = int(N_val)
    fuhao = int(fuhao_val)
    
    # 基本参数
    pc = np.pi * mn
    pn = np.pi / pc
    a = 1.0 / pn
    q = pc / 4
    r = N * mn / 2 / np.cos(beta)
    
    # 刀具坐标系齿面方程
    sin_an = np.sin(an)
    cos_an = np.cos(an)
    tan_an = np.tan(an)
    sin_gama = np.sin(gama)
    cos_gama = np.cos(gama)
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    
    xc = (l * cos_an - a) * cos_gama + sin_gama * (
        -1 * fuhao * (l * sin_an - a * tan_an - q) * sin_beta + u * cos_beta)
    yc = fuhao * cos_beta * (l * sin_an - q - a * tan_an) + u * sin_beta
    zc = -sin_gama * (l * cos_an - a) + cos_gama * (
        -1 * fuhao * sin_beta * (l * sin_an - q - a * tan_an) + u * cos_beta)
    
    # 法向量（常数）
    nxc = -cos_an * sin_beta * sin_gama - 1 * fuhao * sin_an * cos_gama
    nyc = cos_beta * cos_an
    nzc = -cos_an * sin_beta * cos_gama + fuhao * sin_gama * sin_an
    
    # 啮合转角
    f1 = (nxc * yc - nyc * xc) / (nxc * r)
    
    # 齿轮坐标系中的点
    x1 = xc * sp.cos(f1) - yc * sp.sin(f1) + r * (sp.cos(f1) + f1 * sp.sin(f1))
    y1 = xc * sp.sin(f1) + yc * sp.cos(f1) + r * (sp.sin(f1) - f1 * sp.cos(f1))
    z1 = zc
    
    # 法向量在齿轮坐标系中
    nxj1 = nxc * sp.cos(f1) - nyc * sp.sin(f1)
    nyj1 = nxc * sp.sin(f1) + nyc * sp.cos(f1)
    nzj1 = nzc
    
    # 一阶导数
    dl_vec = [sp.diff(x1, l), sp.diff(y1, l), sp.diff(z1, l)]
    du_vec = [sp.diff(x1, u), sp.diff(y1, u), sp.diff(z1, u)]
    
    # 二阶导数
    ddl_vec = [sp.diff(dl_vec[0], l), sp.diff(dl_vec[1], l), sp.diff(dl_vec[2], l)]
    ddu_vec = [sp.diff(du_vec[0], u), sp.diff(du_vec[1], u), sp.diff(du_vec[2], u)]
    du_dl_vec = [sp.diff(du_vec[0], l), sp.diff(du_vec[1], l), sp.diff(du_vec[2], l)]
    
    # 法向量
    nd_vec = [nxj1, nyj1, nzj1]
    
    # 第一基本形式系数
    E1 = sum(du_vec[i] * du_vec[i] for i in range(3))
    F1 = sum(du_vec[i] * dl_vec[i] for i in range(3))
    G1 = sum(dl_vec[i] * dl_vec[i] for i in range(3))
    
    # 第二基本形式系数
    N11 = sum(ddl_vec[i] * nd_vec[i] for i in range(3))
    L1 = sum(ddu_vec[i] * nd_vec[i] for i in range(3))
    M1 = sum(du_dl_vec[i] * nd_vec[i] for i in range(3))
    
    # MATLAB中的求解方法：W1*e^2 + Q1*e + J1 = 0
    W1 = L1 * F1 - M1 * E1
    Q1 = L1 * G1 - N11 * E1
    J1 = M1 * G1 - N11 * F1
    
    # 创建可调用函数
    W1_func = sp.lambdify((l, u), W1, 'numpy')
    Q1_func = sp.lambdify((l, u), Q1, 'numpy')
    J1_func = sp.lambdify((l, u), J1, 'numpy')
    E1_func = sp.lambdify((l, u), E1, 'numpy')
    F1_func = sp.lambdify((l, u), F1, 'numpy')
    L1_func = sp.lambdify((l, u), L1, 'numpy')
    M1_func = sp.lambdify((l, u), M1, 'numpy')
    
    return W1_func, Q1_func, J1_func, E1_func, F1_func, L1_func, M1_func


def calculate_curvature_sympy(gama, beta, an, mn, N, fuhao, u0, l0, xt=0):
    """
    使用SymPy符号计算曲率（精确匹配MATLAB qulvjisuan.m）
    
    Args:
        gama: 节锥角 (rad)
        beta: 螺旋角 (rad)  
        an: 法向压力角 (rad)
        mn: 法向模数
        N: 齿数
        fuhao: 左/右齿面符号 (+1左, -1右)
        u0, l0: 接触点处的曲面参数
        xt: 变位系数（未使用，保持接口兼容）
        
    Returns:
        k1, k2: 两个主曲率
    """
    try:
        # 获取缓存的计算函数
        W1_func, Q1_func, J1_func, E1_func, F1_func, L1_func, M1_func = \
            _get_curvature_function(gama, beta, an, mn, N, fuhao)
        
        # 在接触点求值
        W1 = float(W1_func(l0, u0))
        Q1 = float(Q1_func(l0, u0))
        J1 = float(J1_func(l0, u0))
        E1 = float(E1_func(l0, u0))
        F1 = float(F1_func(l0, u0))
        L1 = float(L1_func(l0, u0))
        M1 = float(M1_func(l0, u0))
        
        # 求解二次方程 W1*e^2 + Q1*e + J1 = 0
        if abs(W1) < 1e-12:
            if abs(Q1) > 1e-12:
                dudl1 = -J1 / Q1
                dudl2 = dudl1
            else:
                return 0.01, 0.01
        else:
            discriminant = Q1**2 - 4 * W1 * J1
            if discriminant < 0:
                return 0.01, 0.01
            sqrt_disc = np.sqrt(discriminant)
            dudl1 = (-Q1 + sqrt_disc) / (2 * W1)
            dudl2 = (-Q1 - sqrt_disc) / (2 * W1)
        
        # 计算曲率 (MATLAB公式)
        denom1 = E1 * dudl1 + F1
        denom2 = E1 * dudl2 + F1
        
        if abs(denom1) > 1e-12:
            k1 = (L1 * dudl1 + M1) / denom1
        else:
            k1 = 0.01
            
        if abs(denom2) > 1e-12:
            k2 = (L1 * dudl2 + M1) / denom2
        else:
            k2 = 0.01
        
        return float(k1), float(k2)
        
    except Exception as e:
        print(f"SymPy曲率计算警告: {e}")
        return 0.01, 0.01


def find_contact_point(gama, beta, an, mn, N, fuhao, xt, zmin, zmax):
    """寻找齿面接触点参数 (l0, u0)"""
    pc = np.pi * mn
    pn = np.pi / pc
    a = 1.0 / pn
    q = pc / 4
    r = N * mn / 2 / np.cos(beta)
    z_mid = (zmin + zmax) / 2
    
    def equations(vars):
        l, u = vars
        
        xc = (l * np.cos(an) - a) * np.cos(gama) + np.sin(gama) * (
            -1 * fuhao * (l * np.sin(an) - a * np.tan(an) - q) * np.sin(beta) + u * np.cos(beta))
        yc = fuhao * np.cos(beta) * (l * np.sin(an) - q - a * np.tan(an)) + u * np.sin(beta)
        zc = -np.sin(gama) * (l * np.cos(an) - a) + np.cos(gama) * (
            -1 * fuhao * np.sin(beta) * (l * np.sin(an) - q - a * np.tan(an)) + u * np.cos(beta))
        
        nxc = -np.cos(an) * np.sin(beta) * np.sin(gama) - 1 * fuhao * np.sin(an) * np.cos(gama)
        nyc = np.cos(beta) * np.cos(an)
        
        f = (nxc * yc - nyc * xc) / (nxc * r)
        
        x1 = xc * np.cos(f) - yc * np.sin(f) + r * (np.cos(f) + f * np.sin(f))
        y1 = xc * np.sin(f) + yc * np.cos(f) + r * (np.sin(f) - f * np.cos(f))
        z1 = zc
        
        eq1 = z1 - z_mid
        eq2 = r - np.sqrt(x1**2 + y1**2)
        
        return [eq1, eq2]
    
    l_init = mn / np.cos(an) / 2
    u_init = z_mid
    
    try:
        solution = fsolve(equations, [l_init, u_init], full_output=True)
        return float(solution[0][0]), float(solution[0][1])
    except:
        return l_init, u_init


def calculate_hertz_contact_stress(params):
    """
    计算赫兹接触应力（纯Python实现，使用SymPy符号计算）
    
    Args:
        params: dict, 齿轮参数和材料参数
            
    Returns:
        dict: {valid, a_len, b_width, Pmax, k1_1, k1_2, k2_1, k2_2, ...}
    """
    try:
        # 提取参数
        E = params.get('E', 210000)
        nu = params.get('nu', 0.3)
        P = params.get('P', 450)
        n1 = params.get('n1', 2500)
        
        N1 = params['N1']
        N2 = params['N2']
        gama1 = params['gama1']
        gama2 = params['gama2']
        beta1 = params['beta1']
        beta2 = params.get('beta2', -beta1)
        xt1 = params.get('xt1', 0)
        xt2 = params.get('xt2', 0)
        b1 = params['b1']
        b2 = params['b2']
        mn = params['mn']
        an = params['an']
        sigma_rad = params.get('sigma_rad', 0)
        fuhao = params.get('fuhao', -1)
        
        zmin1 = params.get('zmin1', -b1/2)
        zmax1 = params.get('zmax1', b1/2)
        zmin2 = params.get('zmin2', -b2/2)
        zmax2 = params.get('zmax2', b2/2)
        
        # 等效弹性模量
        E_star = E / (2 * (1 - nu ** 2))
        
        # 端面压力角和基圆螺旋角
        alpha_t = np.arctan(np.tan(an) / np.cos(beta1))
        beta_b = np.arctan(np.tan(beta1) * np.cos(alpha_t))
        
        # 法向力 - 优先使用直接传入的Fn，否则从P和n1计算
        Fn = params.get('Fn', None)
        
        if Fn is None:
            # 从功率和转速计算力
            drive_gear = params.get('drive_gear', 1)
            
            if drive_gear == 1:
                d_drive = mn * (N1 + 2*xt1)
                beta_drive = beta1
            else:
                d_drive = mn * (N2 + 2*xt2)
                beta_drive = abs(beta2)
            
            dt_drive = d_drive / np.cos(beta_drive)
            
            T1 = 9549 * P / n1
            Ft = 2000 * T1 / dt_drive
            Fn = Ft / (np.cos(an) * np.cos(beta_b))
        
        # 找到接触点
        l10, u10 = find_contact_point(gama1, beta1, an, mn, N1, fuhao, xt1, zmin1, zmax1)
        l20, u20 = find_contact_point(gama2, beta2, an, mn, N2, fuhao, xt2, zmin2, zmax2)
        
        # 使用SymPy符号计算曲率
        k11, k12 = calculate_curvature_sympy(gama1, beta1, an, mn, N1, fuhao, u10, l10, xt1)
        k21, k22 = calculate_curvature_sympy(gama2, beta2, an, mn, N2, fuhao, u20, l20, xt2)
        
        # 齿向修形曲率
        delta1 = params.get('delta1', 0)
        L1 = params.get('L1', b1 * 2)
        deltak1 = 2 * delta1 / (L1**2/4 + delta1**2) if delta1 > 0 else 0
        
        delta2 = params.get('delta2', 0)
        L2 = params.get('L2', b2 * 2)
        deltak2 = 2 * delta2 / (L2**2/4 + delta2**2) if delta2 > 0 else 0
        
        # 取曲率绝对值的最大值作为齿廓曲率
        kai11 = max(abs(k11), abs(k12))
        kai12 = deltak1
        kai21 = max(abs(k21), abs(k22))
        kai22 = deltak2
        
        # 综合曲率
        sum_K = 0.5 * (kai11 + kai12 + kai21 + kai22)
        
        if sum_K <= 0 or np.isnan(sum_K):
            return {'valid': False, 'Pmax': np.nan, 'error': '曲率计算异常'}
        
        # 曲率差
        diff_K = 0.5 * np.sqrt((kai11 - kai12)**2 + (kai21 - kai22)**2 +
                               2*(kai11 - kai12)*(kai21 - kai22)*np.cos(2*sigma_rad))
        
        A = 0.5 * (sum_K - diff_K)
        B = 0.5 * (sum_K + diff_K)
        
        # v9：A <= 0 时为线接触，点接触 Hertz 模型不适用，直接判为无效（原来用 0.4/0.6 的兜底值）
        if A <= 0 or B <= 0:
            return {'valid': False, 'Pmax': np.nan, 'error': '线接触，点接触模型不适用'}

        # 求解接触椭圆偏心率
        def hertz_equation(k):
            m_sq = 1 - k**2
            K_int = ellipk(m_sq)
            E_int = ellipe(m_sq)

            lhs = (K_int - E_int) / (E_int / k**2 - K_int)
            return lhs - (A / B)

        # v9：短长轴比 k 的下限由 0.01 降到 K_MIN，并用区间法求根（左端随 k 单调增加）。
        # 原来 k 被截断在 0.01，FPD 角较小时 Pmax、a_len 为失真的常数。
        # A/B 小于 k = K_MIN 对应的值时，接触椭圆过于细长，判为超出点接触模型的适用范围。
        if hertz_equation(K_MIN) > 0:
            return {'valid': False, 'Pmax': np.nan, 'error': '接触椭圆过于细长，超出点接触模型适用范围'}
        k_val = brentq(hertz_equation, K_MIN, 1 - 1e-9, xtol=1e-14, rtol=1e-12)
        
        # 计算椭圆积分
        m_sq = 1 - k_val**2
        E_int = ellipe(m_sq)
        
        # 赫兹系数
        m = ((2 * E_int) / (np.pi * k_val**2))**(1/3)
        n = k_val * m
        
        # 接触椭圆尺寸
        common_factor = np.cbrt(3 * Fn / (2 * E_star * sum_K))
        a_len = m * common_factor
        b_width = n * common_factor
        
        # 最大接触应力
        if a_len > 0 and b_width > 0:
            Pmax = (3 * Fn) / (2 * np.pi * a_len * b_width)
        else:
            Pmax = np.nan
        
        return {
            'valid': True,
            'a_len': float(a_len),
            'b_width': float(b_width),
            'Pmax': float(Pmax),
            'k1_1': float(kai11),
            'k1_2': float(kai12),
            'k2_1': float(kai21),
            'k2_2': float(kai22),
            'Fn': float(Fn),
            'sum_K': float(sum_K)
        }
        
    except Exception as e:
        return {'valid': False, 'Pmax': np.nan, 'error': str(e)}


def calculate_contact_stress_simple(gear_params, material_params):
    """简化版接触应力计算接口"""
    params = {**gear_params, **material_params}
    result = calculate_hertz_contact_stress(params)
    return result.get('Pmax', np.nan) if result.get('valid', False) else np.nan


if __name__ == "__main__":
    # 测试
    test_params = {
        'P': 1940, 'n1': 2450,
        'N1': 26, 'N2': 79, 'mn': 6,
        'gama1': 5.505847 * np.pi / 180,
        'gama2': 2.578415 * np.pi / 180,
        'beta1': -24.5 * np.pi / 180,
        'beta2': 24.5 * np.pi / 180,
        'an': 20 * np.pi / 180,
        'xt1': 0, 'xt2': 0,
        'b1': 188, 'b2': 180,
        'E': 210000, 'nu': 0.3,
        'fuhao': -1,
        'sigma_rad': -7.8516 / 180 * np.pi,
        'zmin1': -50, 'zmax1': 50,
        'zmin2': -50, 'zmax2': 50,
    }
    
    print("=" * 50)
    print("接触应力计算测试 (纯Python + SymPy符号计算)")
    print("=" * 50)
    
    import time
    
    # 第一次计算（需要符号微分）
    start = time.time()
    result = calculate_hertz_contact_stress(test_params)
    elapsed1 = time.time() - start
    
    print(f"\n第1次计算（符号微分）: {elapsed1*1000:.2f} ms")
    print(f"计算有效: {result.get('valid')}")
    if result.get('valid'):
        print(f"接触椭圆长半轴 a: {result.get('a_len'):.4f} mm")
        print(f"接触椭圆短半轴 b: {result.get('b_width'):.4f} mm")
        print(f"最大接触应力 Pmax: {result.get('Pmax'):.2f} MPa")
        print(f"齿轮1曲率 k1: {result.get('k1_1'):.6f}")
        print(f"齿轮2曲率 k2: {result.get('k2_1'):.6f}")
    else:
        print(f"计算失败: {result.get('error')}")
    
    # 第二次计算（函数已缓存）
    start = time.time()
    result = calculate_hertz_contact_stress(test_params)
    elapsed2 = time.time() - start
    
    print(f"\n第2次计算（函数缓存）: {elapsed2*1000:.2f} ms")
    print(f"加速比: {elapsed1/elapsed2:.1f}x")
    
    print("\n" + "=" * 50)
    print("MATLAB预期值: a=37.23, b=1.08, Pmax=1211.79, k11=0.0301, k21=0.00889")
    print("=" * 50)
