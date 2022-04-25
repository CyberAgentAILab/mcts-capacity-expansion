import numpy as np
import time


class DeferredAcceptance(object):
    def __init__(self, student_prefs, college_prefs):
        self.s, self.c = np.shape(student_prefs)
        self.s_rank = np.array(student_prefs)
        self.s_costs = np.argsort(np.array(student_prefs), axis=1)
        self.c_scores = self.s - np.argsort(np.array(college_prefs).T, axis=0)

        self.arange_s = np.arange(self.s)
        self.arange_c = np.arange(self.c)

    def run(self, college_capacities):
        slots = np.array(college_capacities)
        slots[slots > self.s] = self.s
        s_track = np.zeros(self.s, dtype=int)
        rejected = np.ones(self.s, dtype=int)
        proposals = None
        while np.sum(rejected) > 0:
            proposals = self.s_rank[self.arange_s, s_track]

            receptions = np.zeros((self.s, self.c))
            receptions[self.arange_s, proposals] = 1
            prop_scores = self.c_scores * receptions

            sorted_scores = -np.sort(-prop_scores, axis=0)
            nth_max = sorted_scores[slots - 1, self.arange_c]
            rejected = np.sum((prop_scores < nth_max) * receptions, axis=1, dtype=int)

            s_track += rejected
        matches_for_students = proposals.tolist()
        total_cost = np.sum(self.s_costs[self.arange_s, proposals])
        return matches_for_students, None, float(total_cost)


if __name__ == '__main__':
    student_prefs = [
        [1, 2, 3, 0],
        [1, 2, 3, 0],
        [2, 1, 3, 0],
        [2, 1, 3, 0],
        [0, 3, 1, 2],
        [0, 3, 1, 2],
        [0, 3, 1, 2]
    ]
    college_prefs = [
        [0, 1, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 4, 5, 6]
    ]
    college_capacities = [
        1,
        1,
        1,
        4
    ]

    st = time.time()
    da = DeferredAcceptance(student_prefs, college_prefs)
    matches_for_students, _, total_cost = da.run(college_capacities)
    print(time.time() - st)
    print(matches_for_students, total_cost)
    st = time.time()
