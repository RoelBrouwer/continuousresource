import numpy as np

from continuousresource.localsearch.searchspace \
    import SearchSpace
from continuousresource.mathematicalprogramming.linprog \
    import OrderBasedSubProblem


def stresstest(job_props, bool_func):
    """TODO
    """
    initial_list = [
        [e, i] for i in range(len(job_props)) for e in range(2)
    ]

    searchspace = SearchSpace()
    searchspace.generate_initial_solution(
        OrderBasedSubProblem, initial_list, job_props,
        4, 'dummy_problem'
    )
    random_order = searchspace.get_random_order()

    if random_order is not None:
        LP = OrderBasedSubProblem(random_order, job_props, 4, 'test_problem',
                                  'cplex')
        LP.solve()

        with open('lpstress.csv', 'a') as f:
            f.write(f'{len(job_props)};'
                    f'{LP.problem.solve_details.status == "optimal"};'
                    f'{LP.problem.solve_details.time}\n')

    # From every LP solve, we need the following data:
    # - n, the number of jobs;
    # - feasibility status, we only want to count the LPs that were fed
    #   a feasible order.
    # - solve time, from the solver, not Python-timed
    #   Preferably both ticks & secs


def random_instance(n, p, lowerbounds=False):
    """n = number of jobs, p = average available resource"""
    instance = np.empty(shape=(n, 7))
    instance[:, 0] = np.random.uniform(1, 10, n)                 # E_j

    if lowerbounds:
        instance[:, 1] = np.array([                              # P^-_j
            np.random.uniform(0, np.minimum(p - 1, instance[i, 0]))
            for i in range(n)
        ])
    else:
        instance[:, 1] = np.zeros(n)                             # P^-_j

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


def main():
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


if __name__ == "__main__":
    main()
