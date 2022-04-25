import gurobipy as grb
import numpy as np

from ca_algs.deferred_acceptance import DeferredAcceptance


def iqp(student_prefs, college_prefs, college_capacities, budget, college_budgets, start_capacities=None):
    print('==========Run IPQ Algorithm==========')
    num_students = len(student_prefs)
    num_colleges = len(college_prefs)
    ranks_by_students = [[student_prefs[i].index(j) for j in range(num_colleges)] for i in
                         range(num_students)]
    ranks_by_colleges = [[college_prefs[j].index(i) for i in range(num_students)] for j in
                         range(num_colleges)]
    S = []
    for i in range(num_students):
        S.append([])
        for j in range(num_colleges):
            S[i].append([q for q in range(num_colleges) if ranks_by_students[i][q] <= ranks_by_students[i][j]])
    T = []
    for i in range(num_students):
        T.append([])
        for j in range(num_colleges):
            T[i].append([p for p in range(num_students) if ranks_by_colleges[j][p] < ranks_by_colleges[j][i]])

    model = grb.Model()
    model.setParam('TimeLimit', 60 * 60)
    model.setParam('Threads', 1)
    x = []
    for i in range(num_students):
        x.append([model.addVar(vtype=grb.GRB.BINARY, name='x[{}][{}]'.format(i, j)) for j in range(num_colleges)])
    t = [model.addVar(lb=0, ub=budget, vtype=grb.GRB.INTEGER, name='t[{}]'.format(j)) for j in range(num_colleges)]
    if start_capacities is not None:
        da = DeferredAcceptance(student_prefs, college_prefs)
        matches_for_students, _, _ = da.run(start_capacities)
        for i in range(num_students):
            for j in range(num_colleges):
                x[i][j].start = 1 if matches_for_students[i] == j else 0
        for j in range(num_colleges):
            t[j].start = start_capacities[j] - college_capacities[j]
    model.update()

    model.setObjective(
        grb.quicksum(ranks_by_students[i][j] * x[i][j] for j in range(num_colleges) for i in range(num_students)),
        grb.GRB.MINIMIZE
    )
    model.update()

    for i in range(num_students):
        model.addConstr(grb.quicksum(x[i][j] for j in range(num_colleges)) <= 1, name='student constr[{}]'.format(i))
    for j in range(num_colleges):
        model.addConstr(grb.quicksum(x[i][j] for i in range(num_students)) <= college_capacities[j] + t[j], name='college constr[{}]'.format(j))
    model.addConstr(grb.quicksum(t[j] for j in range(num_colleges)) <= budget)
    for i in range(num_students):
        for j in range(num_colleges):
            model.addConstr((t[j] + college_capacities[j]) * (1 - grb.quicksum(x[i][q] for q in S[i][j]))
                            <= grb.quicksum(x[p][j] for p in T[i][j]), name='matching constr[{}][{}]'.format(i, j))
    if not (np.array(college_budgets) >= budget).all():
        for j in range(num_colleges):
            model.addConstr(t[j] <= college_budgets[j], name='t constr[{}]'.format(j))
    model.update()

    model.optimize()

    expanded_capacities = [j for j in college_capacities]
    best_cost = np.inf
    best_x = [[0 for _ in range(num_colleges)] for _ in range(num_students)]
    best_t = [0 for _ in range(num_colleges)]
    if model.Status in [grb.GRB.OPTIMAL, grb.GRB.TIME_LIMIT]:
        for i in range(num_students):
            for j in range(num_colleges):
                best_x[i][j] = x[i][j].X
        for j in range(num_colleges):
            expanded_capacities[j] += int(t[j].X)
            best_t[j] = t[j].X
        best_cost = model.ObjVal
        print('run time={}'.format(model.Runtime))
        print('reached time limit={}'.format(model.Status == grb.GRB.TIME_LIMIT))
        print('gap={}'.format(model.MIPGap))
        print('node count={}'.format(model.NodeCount))
        print('x={}'.format(np.array(best_x)))
        print('t={}'.format(np.array(best_t)))
        print('objective value={}'.format(best_cost))

    return {
        'expanded_capacities': expanded_capacities,
        'best_cost': best_cost,
        'x': best_x,
        't': best_t,
        'run_time': model.Runtime,
        'gap': model.MIPGap,
        'node_count': model.NodeCount,
        'reached_time_limit': model.Status == grb.GRB.TIME_LIMIT
    }
