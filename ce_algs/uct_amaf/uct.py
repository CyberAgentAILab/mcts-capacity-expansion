import numpy as np
import time

from collections import defaultdict
from ca_algs.deferred_acceptance import DeferredAcceptance
from ce_algs.uct_amaf.capacity_expansion_game import CapacityExpansionGame


class UCTAMAF:
    def __init__(self, depth, exploration_weight=1):
        self.exploration_weight = exploration_weight
        self.node_map = [dict() for _ in range(depth + 1)]
        self.best_history = None
        self.best_history_reward = -np.inf

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node.children is None:
            return node.find_random_child()

        def score(n):
            if n.fully_explored:
                return n.best_reward
            if n.N == 0:
                return float("-inf")  # avoid unseen moves
            return n.Q / n.N  # average reward

        return max(node.children, key=score)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node.children is None or node.terminal:
                # node is either unexplored or terminal
                return path
            # explore first the child node that has never been explored
            unexplored = {nd for nd in node.children if nd.N == 0}
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            # if all child nodes have been explored once, the child node is selected according to UCB values
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node.children is not None:
            return  # already expanded
        # node.children = node.find_children()
        node.children = node.find_children(self.node_map)
        for n in node.children:
            n.parents.add(node)
            self.node_map[len(n.history)][n.history] = n

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        while True:
            if node.is_terminal():
                reward = node.reward()
                if self.best_history_reward < reward:
                    self.best_history_reward = reward
                    self.best_history = node.history
                return reward
            node = node.find_random_child()

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            node.N += 1
            node.Q += reward
            fully_explored = True
            for n in node.children:
                fully_explored = fully_explored & n.fully_explored
                if not fully_explored:
                    break
            if fully_explored:
                node.fully_explored = True
                if len(node.children) == 0:
                    node.best_reward = reward
                else:
                    node.best_reward = max(node.children, key=lambda n: n.best_reward).best_reward
        # print('ba')
        self._rec_backpropagate({path[-1]: 1}, reward)

    def _rec_backpropagate(self, counter, reward):
        for node, count in counter.items():
            # print('update node', node.history, count)
            self._update_node(node, reward, count)
        parent_counter = defaultdict(int)
        for node, count in counter.items():
            for parent in node.parents:
                parent_counter[parent] += count
        if len(parent_counter) > 0:
            self._rec_backpropagate(parent_counter, reward)

    def _update_node(self, node, reward, count):
        node.N += count
        node.Q += reward * count
        fully_explored = True
        for n in node.children:
            fully_explored = fully_explored & n.fully_explored
            if not fully_explored:
                break
        if fully_explored:
            node.fully_explored = True
            if len(node.children) == 0:
                node.best_reward = reward
            else:
                node.best_reward = max(node.children, key=lambda n: n.best_reward).best_reward

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"
        log_N_vertex = np.log(node.N)

        def uct(n):
            "Upper confidence bound for trees"
            return (n.Q / n.N + self.exploration_weight * np.sqrt(log_N_vertex / n.N)) * (1 - n.fully_explored)

        return max(node.children, key=uct)


def uct_amaf(student_prefs, college_prefs, college_capacities, budget, college_budgets):
    return _uct_amaf(student_prefs, college_prefs, college_capacities, budget, college_budgets, UCTAMAF)


def _uct_amaf(student_prefs, college_prefs, college_capacities, budget, college_budgets, alg):
    print('==========Run UCT AMAF==========')
    da = DeferredAcceptance(student_prefs, college_prefs)
    _, _, original_cost = da.run(college_capacities)
    tree = alg(budget, exploration_weight=np.sqrt(0.002))
    game = CapacityExpansionGame(da, student_prefs, college_prefs, college_capacities, budget, college_budgets,
                                 original_cost, [], False)
    tree.node_map[0][tuple([])] = game

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
        best_expanded_capacities[idx] += 1
    _, _, best_cost = da.run(best_expanded_capacities)
    print('run time={}'.format(run_time))
    print('Total cost for expanded capacities {}: {}'.format(best_expanded_capacities, best_cost))

    return {
        'expanded_capacities': best_expanded_capacities,
        'best_cost': best_cost,
        'run_time': run_time
    }, log
