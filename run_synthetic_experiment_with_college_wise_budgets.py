import argparse
import numpy as np

from ca_algs.deferred_acceptance import DeferredAcceptance
from ce_algs import *
from runner import utils
from runner.runner import make_synthetic_instance, run_algs


def run_exp(num_trials, num_students, num_colleges, budget, correlation, load_seed, algs):
    for n in range(num_trials):
        print('==========Run {}-th trial=========='.format(n))
        base_path = 'log/synthetic_with_college_wise_budgets/num_students{}_num_colleges{}_budget{}_correlation{:.1f}/trial{}'\
            .format(num_students, num_colleges, budget, correlation, n)

        # set random seed
        seed = utils.get_seed(base_path, load_seed)
        rng = np.random.RandomState(seed)

        # create instance
        student_prefs, college_prefs, college_capacities, college_budgets = \
            make_synthetic_instance(num_students, num_colleges, budget, correlation, True, rng)

        # save created instance setting
        da = DeferredAcceptance(student_prefs, college_prefs)
        _, _, cost = da.run(college_capacities)
        utils.save_setting(base_path, seed=seed, student_prefs=student_prefs, college_prefs=college_prefs,
                           college_capacities=college_capacities, budget=budget, correlation=correlation, cost=cost)
        print('Instance:\nstudent_prefs {},\ncollege_prefs {},\ncollege_capacities {},\nbudget {},\ncollege budgets {}'
              .format(np.array(student_prefs), np.array(college_prefs), college_capacities, budget, college_budgets))

        # run each algorithm
        run_algs(student_prefs, college_prefs, college_capacities, budget, college_budgets, cost, algs, base_path)


def main():
    parser = argparse.ArgumentParser(description='Main script for synthetic data experiments with college-wise budgets')
    parser.add_argument('--num_students', type=int, required=True, help='number of students')
    parser.add_argument('--num_colleges', type=int, required=True, help='number of colleges')
    parser.add_argument('--budget', type=int, required=True, help='budget for the extra spots to allocate')
    parser.add_argument('--correlation', type=float, default=0.0, help='correlation of student preferences')
    parser.add_argument('--num_trials', type=int, default=10, help='number of trials to run experiments')
    parser.add_argument('--load_seed', action='store_true', help='whether to load a random seed')
    args = parser.parse_args()

    # define algorithms
    algs = [
        greedy,
        lp_heuristic,
        # iqp,
        agg_lin,
        uct_iterative,
        uct_amaf,
        uct_iterative_priority,
        uct_iterative_priority_random,
        uct_iterative_priority_popularity,
        uct_iterative_priority_envy,
        # uct_batch,
        uct_batch_popularity,
        uct_batch_envy,
        uct_batch_random,
    ]

    # run experiments
    print('==========Run experiment over {} trials=========='.format(args.num_trials))
    run_exp(args.num_trials, args.num_students, args.num_colleges, args.budget, args.correlation, args.load_seed, algs)


if __name__ == '__main__':
    main()
