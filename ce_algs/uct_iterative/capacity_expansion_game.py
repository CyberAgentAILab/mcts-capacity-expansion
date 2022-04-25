from random import choice


class CapacityExpansionGame(object):
    def __init__(self, da, student_prefs, college_prefs, college_capacities, budget, college_budgets, original_cost, history, terminal):
        self.student_prefs = student_prefs
        self.college_prefs = college_prefs
        self.college_capacities = college_capacities
        self.num_colleges = len(college_prefs)
        self.depth = budget
        self.college_budgets = college_budgets
        self.base_cost = original_cost
        self.history = [h for h in history]
        self.terminal = terminal
        self.da = da
        self.actions = [i for i in self.da.arange_c if history.count(i) + 1 <= college_budgets[i]]
        self.N = 0
        self.Q = 0
        self.children = None
        self.fully_explored = False
        self.best_reward = None

    def find_children(self):
        if self.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        return {
            self.make_move(i) for i in self.actions
        }

    def find_random_child(self):
        if self.terminal:
            return None  # If the game is finished then no moves can be made
        return self.make_move(choice(self.actions))

    def reward(self):
        if not self.terminal:
            raise RuntimeError(f"reward called on nonterminal board {self}")
        new_capacities = [self.college_capacities[j] for j in range(self.num_colleges)]
        for idx in self.history:
            new_capacities[idx] += 1
        _, _, total_cost = self.da.run(new_capacities)
        return (self.base_cost - total_cost) / self.base_cost

    def is_terminal(self):
        return self.terminal

    def make_move(self, index):
        history = self.history + [index]
        is_terminal = len(history) == self.depth
        return CapacityExpansionGame(self.da, self.student_prefs, self.college_prefs, self.college_capacities,
                                     self.depth, self.college_budgets, self.base_cost, history, is_terminal)

    def __hash__(self):
        return hash(tuple(self.history))
