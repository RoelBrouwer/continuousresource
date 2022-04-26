import os
import time

from continuousresource.simulatedannealing.utils \
    import sanitize_simulated_annealing_params
from continuousresource.simulatedannealing.searchspace \
    import SearchSpaceHillClimb


def simulated_annealing(search_space, params=None):
    """Routine performing Simulated Annealing local search.
    The search stops when one of the following three conditions is met:
        - No candidate solution was accepted after trying 200 options in
          a single iteration, or;
        - The percentage of accepted solutions was under 2% for over 10%
          of the iterations of a single alfa-period, or;
        - The total number of iterations exceeds the `cutoff` parameter.

    Parameters
    ----------
    search_space : SearchSpace
        Search space object that this simulated annealing run will be
        performed on.
    params : Dict
        Dictionary containing parameters for the simulated annealing run,
        with the following keys:
            - `initial_temperature` (float): Initial value for the
              temperature of the annealing process.
            - `alfa` (float): Multiplication factor for updating the
              temperature.
            - `alfa_period` (int): Number of iterations to go through
              before updating the temperature.
            - `cutoff` (int): Maximum number of iterations to run the
              simulated annealing process.

    Returns
    -------
    int
        Number of iterations performed
    SearchSpaceState
        Best found solution.
    """
    # Initialization
    sanitized_parameters = sanitize_simulated_annealing_params(params)
    alfa = sanitized_parameters['alfa']
    alfa_period = sanitized_parameters['alfa_period']
    cutoff = sanitized_parameters['cutoff']
    temperature = sanitized_parameters['initial_temperature']
    iters = cutoff

    # Main loop
    for i in range(cutoff):
        # Select a random neighbor
        fails, new_state = search_space.get_neighbor(temperature)

        if new_state is None:
            # If we were unable to accept a candidate solution from 200
            # options, we give up.
            iters = i + 1
            break

        # Update temperature for next iteration block
        if i % alfa_period:
            temperature = temperature * alfa

    # Return solution
    return iters, search_space.best


def simulated_annealing_verbose(search_space, params=None, output_dir=None):
    """Routine performing Simulated Annealing local search.
    The search stops when one of the following three conditions is met:
        - No candidate solution was accepted after trying 200 options in
          a single iteration, or;
        - The percentage of accepted solutions was under 2% for over 10%
          of the iterations of a single alfa-period, or;
        - The total number of iterations exceeds the `cutoff` parameter.

    Parameters
    ----------
    search_space : SearchSpace
        Search space object that this simulated annealing run will be
        performed on.
    params : Dict
        Dictionary containing parameters for the simulated annealing run,
        with the following keys:
            - `initial_temperature` (float): Initial value for the
              temperature of the annealing process.
            - `alfa` (float): Multiplication factor for updating the
              temperature.
            - `alfa_period` (int): Number of iterations to go through
              before updating the temperature.
            - `cutoff` (int): Maximum number of iterations to run the
              simulated annealing process.
    output_dir : string
        Directory to put the verbose progression report.

    Returns
    -------
    int
        Number of iterations performed
    SearchSpaceState
        Best found solution.
    """
    if output_dir is None:
        output_dir = os.getcwd()

    with open(os.path.join(output_dir,
                           "iterations.csv"), "w") as csv:
        added_header = ""
        if search_space.current.model.with_slack:
            added_header = ";slack"

        csv.write(
            f'#;time;best_score;curr_score;rejected{added_header}\n'
        )

        # Initialization
        sanitized_parameters = sanitize_simulated_annealing_params(params)
        alfa = sanitized_parameters['alfa']
        alfa_period = sanitized_parameters['alfa_period']
        cutoff = sanitized_parameters['cutoff']
        temperature = sanitized_parameters['initial_temperature']
        iters = cutoff
        start_time = time.perf_counter()

        # Main loop
        for i in range(cutoff):
            # Select a random neighbor
            fails, new_state = search_space.get_neighbor(temperature)

            if new_state is None:
                # If we were unable to accept a candidate solution from 200
                # options, we stop.
                iters = i + 1

                slack_string = ""
                if search_space.current.model.with_slack:
                    total_slack = 0
                    for (label, value, weight) in search_space.current.slack:
                        total_slack += value * weight
                    slack_string = f";{total_slack}"

                csv.write(
                    f'{i};{time.perf_counter() - start_time:0.2f};'
                    f'{search_space.best.score};{search_space.current.score};'
                    f'{fails}{slack_string}\n'
                )
                break

            # Update temperature for next iteration block
            if i % alfa_period:
                temperature = temperature * alfa

            slack_string = ""
            if search_space.current.model.with_slack:
                total_slack = 0
                for (label, value, weight) in search_space.current.slack:
                    total_slack += value * weight
                slack_string = f";{total_slack}"

            csv.write(
                f'{i};{time.perf_counter() - start_time:0.2f};'
                f'{search_space.best.score};{search_space.current.score};'
                f'{fails}{slack_string}\n'
            )

    # Return solution
    return iters, search_space.best


def variable_neighborhood_descent(search_space, params=None):
    """Routine performing Variable Neighborhood Descent.

    Parameters
    ----------
    search_space : SearchSpace
        Search space object that this simulated annealing run will be
        performed on.
    params : Dict
        Dictionary containing parameters for the simulated annealing run,
        with the following keys:
            - ...

    Returns
    -------
    int
        Number of iterations performed
    SearchSpaceState
        Best found solution.
    """
    # Initialization
    # sanitized_parameters = sanitize_variable_neighborhood_params(params)
    new_state = search_space.best
    iters = 0

    # Main loop
    while new_state is not None:
        while new_state is not None:
            new_state = search_space.get_neighbor_swap_hc()
            iters += 1
        new_state = search_space.get_neighbor_move_hc()

    # Return solution
    return iters, search_space.best


def iterated_local_search(search_space, params=None, random_walk_length=100, no_restarts=5):
    """Just restarts SA a few times."""
    iters = 0
    
    for _ in range(no_restarts):
        iters1, best = simulated_annealing(search_space, params)
        
        iters += iters1
        # Start random walk from CURRENT (not best) position
        search_space.random_walk(random_walk_length)

    return iters, best
