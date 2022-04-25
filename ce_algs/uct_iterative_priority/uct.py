import numpy as np
import time

from collections import defaultdict
from ca_algs.deferred_acceptance import DeferredAcceptance
from ce_algs.uct_iterative.uct import UCT
from ce_algs.uct_iterative_priority.capacity_expansion_game import CapacityExpansionGame


def uct_iterative_priority(student_prefs, college_prefs, college_capacities, budget, college_budgets):
    def default_order(matches):
        return list(range(len(college_prefs)))
    return _uct_iterative_priority(student_prefs, college_prefs, college_capacities, budget, college_budgets, UCT, default_order)


def uct_iterative_priority_envy(student_prefs, college_prefs, college_capacities, budget, college_budgets):
    def envy_order(matches):
        envy = np.zeros(len(college_prefs))
        for i in range(len(student_prefs)):
            for j in student_prefs[i]:
                if j == matches[i]:
                    break
                envy[j] += 1
        order = np.argsort(envy)
        return order
    return _uct_iterative_priority(student_prefs, college_prefs, college_capacities, budget, college_budgets, UCT, envy_order)


def uct_iterative_priority_popularity(student_prefs, college_prefs, college_capacities, budget, college_budgets):
    def popularity_order(matches):
        student_scores = np.zeros((len(student_prefs), len(college_prefs)))
        for i in range(len(student_prefs)):
            for j in range(len(college_prefs)):
                student_scores[i][student_prefs[i][j]] += j
        mean_preference = np.mean(student_scores, axis=0)
        order = np.argsort(-mean_preference)
        return order
    return _uct_iterative_priority(student_prefs, college_prefs, college_capacities, budget, college_budgets, UCT, popularity_order)


def uct_iterative_priority_random(student_prefs, college_prefs, college_capacities, budget, college_budgets):
    def random_order(matches):
        return np.random.permutation(range(len(college_prefs)))
    return _uct_iterative_priority(student_prefs, college_prefs, college_capacities, budget, college_budgets, UCT, random_order)


def _uct_iterative_priority(student_prefs, college_prefs, college_capacities, budget, college_budgets, alg, order_calculator):
    print('==========Run UCT {} Iterative-tree=========='.format(order_calculator.__name__))
    da = DeferredAcceptance(student_prefs, college_prefs)
    matches, _, original_cost = da.run(college_capacities)
    tree = alg(exploration_weight=np.sqrt(0.002))
    order = order_calculator(matches)
    game = CapacityExpansionGame(da, student_prefs, college_prefs, college_capacities, budget, college_budgets,
                                 original_cost, [], False, order)

    # run rollout
    log = defaultdict(list)
    st_time = time.time()
    for i in range(1000 * budget):
        if i % 100 == 0:
            print('Run {}-th rollout'.format(i))
        if game.fully_explored:
            print('Fully explored! Terminate rollout')
            break
        tree.do_rollout(game)
        log['reward'].append(tree.best_history_reward)
        log['run_time'].append(time.time() - st_time)

    run_time = time.time() - st_time
    best_expanded_capacities = [college_capacities[j] for j in range(len(college_prefs))]
    for idx in tree.best_history:
        best_expanded_capacities[order[idx]] += 1
    _, _, best_cost = da.run(best_expanded_capacities)
    print('run time={}'.format(run_time))
    print('Total cost for expanded capacities {}: {}'.format(best_expanded_capacities, best_cost))

    return {
        'expanded_capacities': best_expanded_capacities,
        'best_cost': best_cost,
        'run_time': run_time
    }, log
