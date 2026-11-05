"""
导出优化前后齿轮系统的全部几何与接触参数（供 FEA 建模和论文表格使用）

用法：
    python export_design.py --summary results_tpe/pinn_w0.8_seed42/summary.json --out logs/design_export.json
"""
import torch  # noqa: F401  （须在 MATLAB 引擎之前加载）
import argparse
import json

import numpy as np

from gear_env_v2 import GearEnvV2, predict_contact_stress_solver
from system import create_example_system

INITIAL = {'gama1': 5.0, 'gama2_gear1': 1.25, 'gama2_gear2': 2.0, 'beta3': 15.0, 'xt3': 0.0}
DEG = ('gama', 'beta', 'an', 'a_', 'kesai', 'deta', 'phi', 'sita')


def evaluate(params):
    base = create_example_system()
    env = GearEnvV2(base)
    env.current_params.update(params)
    g3, p = env.base_params['gear3'], env.current_params
    sys_input = {
        'gear1': {'N2': env.base_params['gear1']['N2'], 'gama2': p['gama2_gear1'], 'b2': env.base_params['gear1']['b2']},
        'gear2': {'N2': env.base_params['gear2']['N2'], 'gama2': p['gama2_gear2'], 'b2': env.base_params['gear2']['b2']},
        'gear3': {k: g3[k] for k in ('mn', 'N1', 'b1', 'Ez', 'han', 'cn', 'an', 'deta_13', 'deta_23', 'offset_direction')},
    }
    sys_input['gear3'].update({'gama1': p['gama1'], 'beta1': p['beta3'], 'xt1': p['xt3'],
                               'ajc_target': g3.get('ajc_target', 320.0)})
    res = env.system.calculate_system(sys_input)
    m = env._calculate_metrics()
    out = {'params': dict(params), 'base': env.base_params, 'min_safety': m['min_safety']}
    for key, pair in (('gear_pair_1', 'pair1'), ('gear_pair_2', 'pair2')):
        d = dict(res[key]['details'])
        # 角度转为度
        for k in list(d):
            if k.startswith(DEG) and k not in ('a_ad1',) and abs(d[k]) < 7:
                d[k + '_deg'] = float(np.degrees(d[k]))
        x = m['contact_inputs_' + pair]
        s = predict_contact_stress_solver(x)
        d.update({'Fn': float(x['Fn']), 'kesai_used_deg': float(x['kesai']),
                  'p_solver': s['Pmax'], 'a_solver': s['a_len'],
                  'p_pgdnn': float(m['Pmax_' + pair]), 'a_pgdnn': float(m['a_len_' + pair])})
        out[pair] = d
    out['O_b'] = out['pair1']['p_solver'] + out['pair2']['p_solver']
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', default='results_tpe/pinn_w0.8_seed42/summary.json')
    parser.add_argument('--out', default='logs/design_export.json')
    args = parser.parse_args()
    best = json.load(open(args.summary, encoding='utf-8'))['best']['params']
    data = {'initial': evaluate(INITIAL), 'optimized': evaluate(best)}
    json.dump(data, open(args.out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=float)
    for k, v in data.items():
        print(k, 'O_b =', round(v['O_b'], 2), 'S_min =', v['min_safety'],
              'p23/p13 =', round(v['pair1']['p_solver'], 2), round(v['pair2']['p_solver'], 2))


if __name__ == '__main__':
    main()
