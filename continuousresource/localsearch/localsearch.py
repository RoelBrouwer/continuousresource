import os
import time

from .utils import sanitize_simulated_annealing_params


def simulated_annealing(search_space, params=None):
    """Routine performing Simulated Annealing local search.

    The search stops when one of the following two conditions is met:
        - No candidate solution was accepted after trying all options
          for every neighborhood operators (check the documentation of
          each neighborhood operator to see what this entails).
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
    prev_score = search_space.current.score

    # Main loop
    for i in range(cutoff):
        # Select a random neighbor
        fails, new_state = search_space.get_neighbor(temperature)

        if new_state is None:
            # If we were unable to accept a candidate solution from all
            # available options, we give up.
            iters = i + 1
            break

        # Update temperature for next iteration block
        if i > 0 and i % alfa_period == 0:
            difference = (search_space.current.score - prev_score) / prev_score
            if difference < 0.001:
                iters = i + 1
                break
            prev_score = search_space.current.score
            temperature = temperature * alfa

    # Return solution
    return iters, search_space.best


def simulated_annealing_verbose(search_space, params=None, output_dir=None):
    """Routine performing Simulated Annealing local search. Produces more
    output than the regulare simulated annealing routine focussing on
    individual iterations and time measurements.

    The search stops when one of the following three conditions is met:
        - No candidate solution was accepted after trying all options
          for every neighborhood operators (check the documentation of
          each neighborhood operator to see what this entails).
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
        prev_score = search_space.current.score
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

            # Update temperature for next iteration block
            if i > 0 and i % alfa_period == 0:
                difference = (search_space.current.score - prev_score) / prev_score
                if difference < 0.001:
                    iters = i + 1
                    break
                prev_score = search_space.current.score
                temperature = temperature * alfa

    # Return solution
    return iters, search_space.best


def variable_neighborhood_descent(search_space):
    """Routine performing Variable Neighborhood Descent.

    Parameters
    ----------
    search_space : SearchSpace
        Search space object that this variable neighborhood search will
        be performed on.

    Returns
    -------
    int
        Number of iterations performed
    SearchSpaceState
        Best found solution.
    """
    # Initialization
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


def iterated_local_search(search_space, params=None, random_walk_length=100,
                          no_restarts=5):
    """Performs Iterated Local Search by performing `no_restarts` runs of
    the simulated annealing routine, and distorting the final solution
    (note: not necessarily the best found solution) by performing a
    random walk of length `random_walk_length` after each run.

    Parameters
    ----------
    search_space : SearchSpace
        Search space object that the simulated annealing runs will be
        performed on.
    params : Dict
        Dictionary containing parameters for the simulated annealing
        runs, with the following keys:
            - `initial_temperature` (float): Initial value for the
              temperature of the annealing process.
            - `alfa` (float): Multiplication factor for updating the
              temperature.
            - `alfa_period` (int): Number of iterations to go through
              before updating the temperature.
            - `cutoff` (int): Maximum number of iterations to run the
              simulated annealing process.
    random_walk_length : int
        Length of the random walk performed after each simulated
        annealing run, distorting the solution to find a new semi-random
        starting point.
    no_restarts : int
        Number of simulated annealing runs to perform.

    Returns
    -------
    int
        Total number of iterations performed across all runs.
    SearchSpaceState
        Best found solution.
    """
    iters = 0

    for _ in range(no_restarts):
        iters1, best = simulated_annealing(search_space, params)

        iters += iters1
        # Start random walk from CURRENT (not best) position
        search_space.random_walk(random_walk_length)

    return iters, best
