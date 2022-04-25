import numpy as np
import time

from ca_algs.deferred_acceptance import DeferredAcceptance


def greedy(student_prefs, college_prefs, college_capacities, budget, college_budgets):
    print('==========Run Greedy Algorithm==========')
    expanded_capacities = [j for j in college_capacities]
    best_cost = np.infty

    da = DeferredAcceptance(student_prefs, college_prefs)
    st_time = time.time()
    for b in range(budget):
        print('===Search allocation of {}-th extra capacity==='.format(b))
        new_costs = [-1 for _ in range(len(college_prefs))]
        for i in range(len(college_prefs)):
            new_capacities = [j for j in expanded_capacities]
            new_capacities[i] += 1
            if new_capacities[i] - college_capacities[i] > college_budgets[i]:
                total_cost = np.inf
            else:
                _, _, total_cost = da.run(new_capacities)
            new_costs[i] = total_cost
            print('Total cost when {}-th extra capacity is allocated to college {}: {}'.format(b, i, total_cost))
            best_cost = min(best_cost, total_cost)
        min_college_id = np.argmin(new_costs)
        expanded_capacities[min_college_id] += 1
        print('{}-th extra capacity is allocated to college {}: new capacities are {}, new cost is {}'
              .format(b, min_college_id, expanded_capacities, best_cost))
    run_time = time.time() - st_time

    return {
        'expanded_capacities': expanded_capacities,
        'best_cost': best_cost,
        'run_time': run_time
    }
