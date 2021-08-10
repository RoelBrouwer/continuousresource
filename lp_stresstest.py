import numpy as np

from linprog import OrderBasedSubProblem
from decomposition import EventOrderMasterProblem


def stresstest(job_props, bool_func):
    """TODO
    """
    initial_list = [
        [e, i] for i in range(len(job_props)) for e in range(2)
    ]

    problem = EventOrderMasterProblem(job_props)

    prec_matrix = problem.compute_precedence_matrix(initial_list, bool_func)
    random_order = problem.get_random_order(initial_list, prec_matrix)

    if random_order is not None:
        LP = OrderBasedSubProblem(random_order, job_props, 4, 'test_problem', 'cplex')
        LP.solve()
        # print(LP.problem.solve_details.status)

        with open('lpstress.csv', 'a') as f:
            f.write(f'{len(job_props)};{LP.problem.solve_details.status == "optimal"};{LP.problem.solve_details.time}\n')
    # TODO: build LP, log results

    # From every LP solve, we need the following data:
    # - n, the number of jobs;
    # - feasibility status, we only want to count the LPs that were fed
    #   a feasible order.
    # - solve time, from the solver, not Python-timed
    #   Preferably both ticks & secs


def random_instance(n, p):
    """n = number of jobs, p = average available resource"""
    instance = np.empty(shape=(n, 7))
    instance[:, 0] = np.random.uniform(1, 10, n)                 # E_j
    instance[:, 1] = np.zeros(n) #np.array([                                  # P^-_j
        #np.random.uniform(0, np.minimum(p - 1, instance[i, 0])) 
        #for i in range(n)
    #])
    instance[:, 2] = np.array([                                  # P^+_j
        np.random.uniform(instance[i, 1], 10) 
        for i in range(n)
    ])
    instance[:, 3] = np.random.uniform(0, 5.5 * (n / p), n)      # r_j
    instance[:, 4] = np.add(                                     # d_j
        instance[:, 3],
        np.array([
            np.random.uniform(2 * instance[i, 0] / instance[i, 2], 30)
            for i in range(n)
        ])
    )
    instance[:, 5] = np.random.uniform(0, 1, n)                  # W_j
    instance[:, 6] = np.zeros(n)                                 # B_j

    return instance

if __name__ == "__main__":
    def redux(i, j, job_properties):
        if (i[1] == j[1] and i[0] > j[0]):
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

    # Create output file
    with open('lpstress.csv', 'w') as f:
        f.write('n;feasibility;solve_time\n')

    test_pairs = [
        # (n, repeats)
        (4, 10),
        (8, 10),
        (16, 10),
        (32, 10),
        (64, 10),
        (128, 10),
        (256, 10),
        (512, 10),
        (1024, 10),
        (2048, 10),
        (4096, 10)
    ]

    for pair in test_pairs:
        for j in range(pair[1]):
            instance = random_instance(pair[0], 4)
            stresstest(instance, redux)