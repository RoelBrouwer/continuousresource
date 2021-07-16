"""Tests for the functions defined in linprog.py.
Currently, manual testing is required, but the structure is set up with
eventual automation/parameterization (with pytest) in mind.
"""
import numpy as np

from linprog import OrderBasedSubProblem


def test_small_instance():
    # General Eventlist
    # eventlist = np.array([[e, i] for i in range(3) for e in range(2)])
    # Feasible eventlist
    eventlist = np.array([
        [0, 2], [0, 1], [1, 2], [1, 1], [0, 0], [1, 0]
    ])
    jobs = np.array([
        # E_j, P^-_j, P^+_j, r_j, d_j, W_j, B_j
        [3, 1, 3, 1, 3, 1, 0],
        [4, 1, 3, 0, 3, 1, 0],
        [3, 1.5, 3, 0, 2, 1, 0]
    ])
    LP = OrderBasedSubProblem(eventlist, jobs, 4, 'test_problem', 'cplex')
    LP.solve()


if __name__ == "__main__":
    test_small_instance()