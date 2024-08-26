import numpy as np

def node_visit_constraints(instance):
    constraints = []

    C = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'c']
    R = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'f']
    N = list(range(instance.N))
    K = list(range(instance.k_e + instance.k_f))
    K_e = list(range(instance.k_e))
    M = instance.M

    # 起点的出车等于终点的回车
    constraints.append(
        sum(instance.x[0, j, k] for j in C + R for k in K) == sum(instance.x[j, instance.O_prime, k] for j in C + R for k in K)
    )

    # 每个客户点只能被访问一次
    for i in C:
        constraints.append(
            sum(instance.x[i, j, k] for j in N if j != i for k in K) == 1
        )

    # 保证每个客户点的进出流平衡
    for j in C:
        for k in K:
            constraints.append(
                sum(instance.x[i, j, k] for i in N if i != j) == sum(instance.x[j, i, k] for i in N if i != j)
            )

    # 每个电动车对每个充电站最多访问一次
    for r in R:
        for k in K_e:
            constraints.append(
                sum(instance.x[i, r, k] for i in N if i != r) <= 1
            )
            constraints.append(
                sum(instance.x[r, j, k] for j in N if j != r) <= 1
            )

    return constraints

def load_balance_constraints(instance):
    constraints = []

    C = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'c']
    R = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'f']
    N = list(range(instance.N))
    E = [(i, j) for i in N for j in N if i != j]
    K = list(range(instance.k_e + instance.k_f))
    K_e = list(range(instance.k_e))
    K_f = list(range(instance.k_e, instance.k_e + instance.k_f))
    M = instance.M
    Q_e = instance.Q_e
    Q_f = instance.Q_f
    q = instance.customer_demand

    for (i, j) in E:
        for k in K_e:
            constraints.append(instance.u[i, j, k] <= Q_e)
        for k in K_f:
            constraints.append(instance.u[i, j, k] <= Q_f)

    for j in C:
        for k in K:
            constraints.append(
                sum(instance.u[i, j, k] for i in N if i != j) - sum(instance.u[j, i, k] for i in N if i != j) + M * (1 - sum(instance.x[i, j, k] for i in N if i != j)) >= q[j]
            )

    for k in K:
        constraints.append(
            sum(instance.u[i, instance.O_prime, k] for i in C + R) == 0
        )

    return constraints

def time_constraints(instance):
    constraints = []

    C = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'c']
    R = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'f']
    N = list(range(instance.N))
    K = list(range(instance.k_e + instance.k_f))
    M = instance.M
    t = instance.travel_time_matrix
    s = instance.service_time
    w = np.zeros_like(s)
    E = [loc['ready_time'] for loc in instance.locations]
    L = [loc['due_date'] for loc in instance.locations]

    for k in K:
        constraints.append(instance.b[0, k] >= E[0])

    for j in C:
        for k in K:
            constraints.append(
                instance.b[0, k] + t[0, j] * instance.x[0, j, k] - M * (1 - instance.x[0, j, k]) <= instance.a[j, k]
            )

    for i in C:
        for j in N:
            if j != i:
                for k in K:
                    constraints.append(
                        instance.a[i, k] + t[i, j] * instance.x[i, j, k] + s[i] + w[i, k] - M * (1 - instance.x[i, j, k]) <= instance.a[j, k]
                    )

    for i in N:
        for k in K:
            constraints.append(
                instance.b[i, k] + t[i, instance.O_prime] * instance.x[i, instance.O_prime, k] - M * (1 - instance.x[i, instance.O_prime, k]) <= L[0]
            )

    for i in C:
        for k in K:
            constraints.append(
                instance.b[i, k] == instance.a[i, k] + s[i] + w[i, k]
            )

    for i in R:
        for k in K_e:
            constraints.append(
                instance.b[i, k] == instance.a[i, k] + M
            )

    for i in C:
        for k in K_e:
            constraints.append(
                E[i] <= instance.a[i, k] <= L[i]
            )

    return constraints

def battery_constraints(instance):
    constraints = []

    C = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'c']
    R = [i for i, loc in enumerate(instance.locations) if loc['type'] == 'f']
    N = list(range(instance.N))
    K_e = list(range(instance.k_e))
    M = instance.M
    L_ijk = instance.L_ijk
    B_star = instance.B_star
    soc_min = instance.soc_min

    for i in N:
        for j in N:
            if j != i:
                for k in K_e:
                    constraints.append(
                        instance.B[i, k] - L_ijk[i, j] * instance.x[i, j, k] + B_star * (1 - instance.x[i, j, k]) >= instance.B[j, k]
                    )

    for k in K_e:
        constraints.append(
            instance.B[0, k] == B_star
        )

    for i in N:
        for k in K_e:
            constraints.append(
                instance.B[i, k] >= B_star * soc_min
            )

    for r in R:
        for k in K_e:
            constraints.append(
                instance.B[r, k] == B_star
            )

    return constraints

def variable_constraints(instance):
    constraints = []

    N = list(range(instance.N))
    K = list(range(instance.k_e + instance.k_f))

    for (i, j) in [(i, j) for i in N for j in N if i != j]:
        for k in K:
            constraints.append(instance.u[i, j, k] >= 0)
            if k in range(instance.k_e):
                constraints.append(instance.f[i, j, k] >= 0)
            constraints.append(instance.x[i, j, k] == 0 or instance.x[i, j, k] == 1)

    return constraints
