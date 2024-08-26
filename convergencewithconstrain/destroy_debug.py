class DestructionOperators:
    def __init__(self, solution, rnd_state):
        self.solution = solution
        self.rnd_state = rnd_state

    def random_removal(self, curr_solution, rnd_state, **kwargs):
        num_customers_to_remove = kwargs.get('num_customers_to_remove')
        if num_customers_to_remove is None:
            raise ValueError("num_customers_to_remove 参数缺失")

        customers = list(self.solution.customers)
        removed_customers = rnd_state.choice(customers, size=num_customers_to_remove, replace=False)
        for customer in removed_customers:
            self.solution.remove_customer(customer)
        return removed_customers

    # 其他破坏算子实现类似，略去重复代码

class InsertionOperators:
    def __init__(self, instance):
        self.instance = instance

    def greedy_insertion(self, curr_solution, rnd_state, **kwargs):
        customers = kwargs.get('customers', [])
        # 插入逻辑
        # 其他修复算子实现类似，略去重复代码
