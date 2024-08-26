import numpy as np

def calculate_scores(instance):
    # 初始化两个集合 E 和 C
    E = []
    C = []

    # 计算电动车和燃油车的初始距离
    d_E = np.linalg.norm(instance.customer_positions - instance.depot_position, axis=1)
    d_C = np.linalg.norm(instance.customer_positions - instance.depot_position, axis=1)

    # 计算电动车的评分
    d_E_min = np.min(d_E)
    d_E_max = np.max(d_E)
    p_E = 11 - 1 + (d_E - d_E_min) / (d_E_max - d_E_min) * 9

    # 计算燃油车的评分
    d_C_min = np.min(d_C)
    d_C_max = np.max(d_C)
    q_min = np.min(instance.customer_demand)
    q_max = np.max(instance.customer_demand)

    lambda_param = 0.5  # 可调参数
    pDist_C = 11 - 1 + (d_C - d_C_min) / (d_C_max - d_C_min) * 9
    pQ = 11 - 1 + (instance.customer_demand - q_min) / (q_max - q_min) * 9
    p_C = lambda_param * pDist_C + (1 - lambda_param) * pQ

    return p_E, p_C, q_min, q_max, lambda_param, pDist_C, pQ

def clustering(instance):
    # 初始化
    p_E, p_C, q_min, q_max, lambda_param, pDist_C, pQ = calculate_scores(instance)
    E, C = [], []
    unassigned_customers = set(range(instance.n))

    while unassigned_customers:
        i_E = max(unassigned_customers, key=lambda i: p_E[i])
        i_C = max(unassigned_customers, key=lambda i: p_C[i])

        if i_E == i_C:
            E.append(i_E)
            unassigned_customers.remove(i_E)
        else:
            if p_E[i_E] > p_C[i_C]:
                E.append(i_E)
                unassigned_customers.remove(i_E)
            else:
                C.append(i_C)
                unassigned_customers.remove(i_C)

        # 重新计算未分配客户的评分
        remaining_customers = list(unassigned_customers)
        if remaining_customers:
            E_positions = instance.customer_positions[E]
            C_positions = instance.customer_positions[C]

            E_barycenter = np.mean(E_positions, axis=0) if E_positions.size else instance.depot_position
            C_barycenter = np.mean(C_positions, axis=0) if C_positions.size else instance.depot_position

            d_E = np.linalg.norm(instance.customer_positions[remaining_customers] - E_barycenter, axis=1)
            d_C = np.linalg.norm(instance.customer_positions[remaining_customers] - C_barycenter, axis=1)

            d_E_min = np.min(d_E)
            d_E_max = np.max(d_E)
            d_C_min = np.min(d_C)
            d_C_max = np.max(d_C)

            p_E[remaining_customers] = 11 - 1 + (d_E - d_E_min) / (d_E_max - d_E_min) * 9
            pDist_C[remaining_customers] = 11 - 1 + (d_C - d_C_min) / (d_C_max - d_C_min) * 9
            pQ[remaining_customers] = 11 - 1 + (instance.customer_demand[remaining_customers] - q_min) / (q_max - q_min) * 9
            p_C[remaining_customers] = lambda_param * pDist_C[remaining_customers] + (1 - lambda_param) * pQ[remaining_customers]

    return E, C
