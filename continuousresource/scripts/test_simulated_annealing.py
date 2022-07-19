import click
import numpy as np

from continuousresource.localsearch.localsearch \
    import SearchSpace, simulated_annealing
from continuousresource.mathematicalprogramming.linprog \
    import OrderBasedSubProblemWithSlack


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    '--temp-init',
    '-t',
    'initial_temperature',
    required=True,
    type=float,
    help="Initial temperature for the simulated annealing process."
)
@click.option(
    '--alfa',
    '-a',
    'alfa',
    default=0.95,
    required=False,
    type=float,
    help="Multiplication factor used to iteratively cool the temperature."
)
@click.option(
    '--period-alfa',
    '-p',
    'alfa_period',
    default=1,
    required=False,
    type=int,
    help="Number of iterations in between two temperature updates."
)
def main(initial_temperature, alfa, alfa_period):
    model_class = OrderBasedSubProblemWithSlack
    # Test instance (very small)
    # J1. E = 6; no bounds, release date or deadline; weight = 1
    # J2. E = 4; no bounds, release date or deadline; weight = 1
    # P = 2
    eventlist = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    # eventlist = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
    jobs = np.array([[6.0, 0, 6, 0, 6, 1, 0], [4.0, 0, 4, 0, 6, 1, 0]])
    resource = 2
    slackpenalties = [2, 2]
    label = "simann_model"

    search_space = SearchSpace()
    search_space.generate_initial_solution(model_class, eventlist, jobs,
                                           resource, slackpenalties, label)

    solution = simulated_annealing(search_space, initial_temperature, alfa,
                                   alfa_period)

    print(f'Objective value of best found solution: {solution.score}.\n')
    print(f'Event order for this solution: {solution.eventorder}\n')
    print(f'Corresponding schedule: {solution.schedule}')


if __name__ == "__main__":
    main()
