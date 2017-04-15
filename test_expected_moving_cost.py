from policy_evaluation import DPSolver

solver = DPSolver()


print(solver.expected_moving_cost(0, 0, 0))

print(solver.expected_moving_cost(0, 0, -1))
print(solver.expected_moving_cost(0, 0, -2))
print(solver.expected_moving_cost(0, 0, -5))
