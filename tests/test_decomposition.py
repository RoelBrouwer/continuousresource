"""Tests for the functions defined in decomposition.py.
Currently, manual testing is required, but the structure is set up with
eventual automation/parameterization (with pytest) in mind.
"""
import numpy as np

from decomposition import EventOrderMasterProblem


def test_precedence_matrix_computation(job_properties, event_list, bool_func):
    problem = EventOrderMasterProblem(job_properties)
    prec_matrix = problem.compute_precedence_matrix(event_list, bool_func)
    print(np.matrix(prec_matrix))


if __name__ == "__main__":
    jobs = np.array([
        # E_j, P^-_j, P^+_j, r_j, d_j, W_j, B_j
        [3, 1, 3, 1, 3, 1, 0],
        [4, 1, 3, 0, 3, 1, 0],
        [3, 1.5, 3, 0, 2, 1, 0]
    ])
    initial_list = [
        [e, i] for i in range(len(jobs)) for e in range(2)
    ]
    # Only the most basic rule (property 1): j goes before i if i and j
    # are linked to the same job, of which i represents the completion,
    # and j represents the start event.

    def func(i, j, job_properties):
        return (i[1] == j[1] and j[0] == 0 and i[0] == 1)

    # Correct in theory, but sensitive to floating point errors. TO FIX.

    def all_redux(i, j, job_properties):
        if (i[1] == j[1] and i[0] == 1 and j[0] == 0):
            return True

        return (
            job_properties[i[1], 3] +                          # r_i
            i[0] *                                             # event_type
            job_properties[i[1], 0] / job_properties[i[1], 2]  # E_i / P^+_i
            >=                                                 # boolean operator
            job_properties[j[1], 4] -                          # deadline
            (1 - j[0]) *                                       # event_type
            job_properties[j[1], 0] / job_properties[j[1], 2]  # E_j / P^+_j
        )

    def all_redux_stable(i, j, job_properties):
        if (i[1] == j[1] and i[0] == 1 and j[0] == 0):
            return True

        a = (
            job_properties[i[1], 3] +                          # r_i
            i[0] *                                             # event_type
            job_properties[i[1], 0] / job_properties[i[1], 2]  # E_i / P^+_i
        )
        b = (
            job_properties[j[1], 4] -                          # deadline
            (1 - j[0]) *                                       # event_type
            job_properties[j[1], 0] / job_properties[j[1], 2]  # E_j / P^+_j
        )
        return a >= b or np.isclose(a, b)

    test_precedence_matrix_computation(jobs, initial_list, all_redux_stable)
