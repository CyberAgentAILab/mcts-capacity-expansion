import numpy as np
import pickle

from runner import utils


def make_synthetic_instance(num_students, num_colleges, budget, correlation, with_college_wise_budget, rng):
    common_scores = rng.random(num_colleges)
    student_scores = rng.random((num_students, num_colleges))
    mixed_scores = correlation * common_scores + (1 - correlation) * student_scores
    student_prefs = np.argsort(mixed_scores).tolist()
    college_prefs = [rng.permutation(np.arange(num_students)).tolist() for _ in range(num_colleges)]
    college_capacities = (rng.multinomial(num_students - num_colleges, np.ones(num_colleges) / num_colleges) + 1).tolist()
    if with_college_wise_budget:
        while True:
            college_budgets = rng.multinomial(rng.randint(budget, budget * num_colleges), np.ones(num_colleges) / num_colleges) + 1
            if (college_budgets < budget).all():
                college_budgets = college_budgets.tolist()
                break
    else:
        college_budgets = [budget] * num_colleges
    return student_prefs, college_prefs, college_capacities, college_budgets


def make_real_data_student_preferences(apply_cluster, regional_cap, rng):
    # make student's preferences
    student_apply = {}  # key: [0,...,regional_cap-1], value: list of colleges applied by student i
    for i in range(regional_cap):
        student_apply[i] = []
    K = len(apply_cluster)
    for j in range(K):
        numapplication = apply_cluster[j]
        ii = 0
        while ii < numapplication:
            i = rng.randint(0, regional_cap)
            if (j not in student_apply[i]) and len(student_apply[i]) < 8:
                student_apply[i].append(j)
                ii += 1
            else:
                pass
    student_preference = np.zeros((regional_cap, K + 1))
    for i in range(regional_cap):
        L = len(student_apply[i])
        for k in range(L):
            student_preference[i, k] = student_apply[i][k]
        student_preference[i, L] = K  # dummy college
        # fill remaining preferences
        tmp = 0
        for k in range(L + 1, K + 1):
            while tmp in student_apply[i]:
                tmp = tmp + 1
            student_preference[i, k] = tmp
            tmp = tmp + 1
    return student_preference.astype(int).tolist()


def make_real_data_instance(rng):
    with open('./data/tokyo.pkl', 'rb') as f:
        data = pickle.load(f)
        college_capacities = data[0].tolist()
        college_budgets = data[1].astype(int).tolist()
        apply_cluster = data[2]
    student_prefs = make_real_data_student_preferences(apply_cluster, np.sum(college_capacities) // 2, rng)
    college_prefs = [rng.permutation(np.arange(len(student_prefs))).tolist() for _ in range(len(college_capacities))]
    return student_prefs, college_prefs, college_capacities, college_budgets


def run_algs(student_prefs, college_prefs, college_capacities, budget, college_budgets, original_cost, algs, save_path):
    best_cost_by_heuristics = np.inf
    best_solution_by_heuristics = None
    # run each algorithm
    for i in range(len(algs)):
        log = None
        if algs[i].__name__ in ['iqp', 'agg_lin', 'non_agg_lin']:
            result = algs[i](student_prefs, college_prefs, college_capacities, budget, college_budgets, best_solution_by_heuristics)
        else:
            results = algs[i](student_prefs, college_prefs, college_capacities, budget, college_budgets)
            if not isinstance(results, dict):
                result = results[0]
                log = results[1]
            else:
                result = results
            if best_cost_by_heuristics > result['best_cost'] and algs[i].__name__ in ['greedy', 'lp_heuristic']:
                best_solution_by_heuristics = result['expanded_capacities']
                best_cost_by_heuristics = result['best_cost']
        result['improvement_rate'] = (original_cost - result['best_cost']) / original_cost
        print('Allocated capacities are {}'.format(np.array(result['expanded_capacities']) - np.array(college_capacities)))
        print('Expanded capacities are {}, new cost is {}, improvement rate is {}'
              .format(result['expanded_capacities'], result['best_cost'], result['improvement_rate']))
        # save result
        utils.save_result('{}/{}'.format(save_path, algs[i].__name__), result, log)
