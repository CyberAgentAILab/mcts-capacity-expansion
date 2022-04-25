import gurobipy as grb
import numpy as np

from ca_algs.deferred_acceptance import DeferredAcceptance


def lp_heuristic(student_prefs, college_prefs, college_capacities, budget, college_budgets):
    print('==========Run LPH Algorithm==========')
    num_students = len(student_prefs)
    num_colleges = len(college_prefs)
    ranks_by_students = [[student_prefs[i].index(j) for j in range(num_colleges)] for i in range(num_students)]

    model = grb.Model()
    model.setParam('TimeLimit', 60 * 60)
    model.setParam('Threads', 1)
    x = []
    for i in range(num_students):
        x.append([model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='x[{}][{}]'.format(i, j)) for j in range(num_colleges)])
    t = [model.addVar(lb=0, ub=budget, vtype=grb.GRB.CONTINUOUS, name='t[{}]'.format(j)) for j in range(num_colleges)]
    model.update()

    model.setObjective(
        grb.quicksum(ranks_by_students[i][j] * x[i][j] for j in range(num_colleges) for i in range(num_students)),
        grb.GRB.MINIMIZE
    )
    model.update()

    for i in range(num_students):
        model.addConstr(grb.quicksum(x[i][j] for j in range(num_colleges)) == 1, name='student constr[{}]'.format(i))
    for j in range(num_colleges):
        model.addConstr(grb.quicksum(x[i][j] for i in range(num_students)) <= college_capacities[j] + t[j], name='college constr[{}]'.format(j))
    model.addConstr(grb.quicksum(t[j] for j in range(num_colleges)) <= budget)
    if not (np.array(college_budgets) >= budget).all():
        for j in range(num_colleges):
            model.addConstr(t[j] <= college_budgets[j], name='t constr[{}]'.format(j))
    model.update()

    model.optimize()

    expanded_capacities = [j for j in college_capacities]
    best_x = [[0 for _ in range(num_colleges)] for _ in range(num_students)]
    best_t = [0 for _ in range(num_colleges)]
    if model.Status in [grb.GRB.OPTIMAL, grb.GRB.TIME_LIMIT]:
        for i in range(num_students):
            for j in range(num_colleges):
                best_x[i][j] = x[i][j].X
        for j in range(num_colleges):
            expanded_capacities[j] += int(t[j].X)
            best_t[j] = t[j].X
        print('run time={}'.format(model.Runtime))
        print('reached time limit={}'.format(model.Status == grb.GRB.TIME_LIMIT))
        print('x={}'.format(np.array(best_x)))
        print('t={}'.format(np.array(best_t)))

    da = DeferredAcceptance(student_prefs, college_prefs)
    _, _, best_cost = da.run(expanded_capacities)
    return {
        'expanded_capacities': expanded_capacities,
        'best_cost': best_cost,
        'x': best_x,
        't': best_t,
        'run_time': model.Runtime,
        'reached_time_limit': model.Status == grb.GRB.TIME_LIMIT
    }
