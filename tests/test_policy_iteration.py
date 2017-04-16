from jackscarrental import PolicyIterationSolver
solver = PolicyIterationSolver()

for ii in range(4):
    solver.policy_evaluation()
    solver.policy_update()

print(solver.policy)

import matplotlib.pylab as plt

plt.subplot(121)
CS = plt.contour(solver.policy, levels=range(-6, 6))
plt.clabel(CS)
plt.xlim([0, 20])
plt.ylim([0, 20])
plt.axis('equal')
plt.xticks(range(21))
plt.yticks(range(21))
plt.grid('on')

plt.subplot(122)
plt.pcolor(solver.value)
plt.colorbar()
plt.axis('equal')

plt.show()

