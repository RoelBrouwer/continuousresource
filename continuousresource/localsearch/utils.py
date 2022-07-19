"""Utility functions that *may* be reused across simulated annealing
implementations."""


import warnings


def sanitize_simulated_annealing_params(params):
    """Check parameter dictionary for missing values and substitute
    defaults where appropriate.

    Parameters
    ----------
    params : Dict
        Dictionary containing parameters for the simulated annealing run.

    Returns
    -------
    Dict
        Sanitized dictionary, with the following keys:
            - `initial_temperature` (float): Initial value for the
              temperature of the annealing process.
            - `alfa` (float): Multiplication factor for updating the
              temperature.
            - `alfa_period` (int): Number of iterations to go through
              before updating the temperature.
            - `cutoff` (int): Maximum number of iterations to run the
              simulated annealing process.
    """
    if params is None:
        params = {}
    sanitized_params = {}

    # Initial temperature
    if 'initial_temperature' in params:
        sanitized_params['initial_temperature'] = \
            float(params['initial_temperature'])
    else:
        warnings.warn(
            "No setting for initial temperature detected. Using default:"
            " 100.0.",
            RuntimeWarning
        )
        sanitized_params['initial_temperature'] = 100.0

    # Alfa
    if 'alfa' in params:
        sanitized_params['alfa'] = \
            float(params['alfa'])
    else:
        warnings.warn(
            "No setting for alfa detected. Using default: 0.95.",
            RuntimeWarning
        )
        sanitized_params['alfa'] = 0.95

    # Alfa period
    if 'alfa_period' in params:
        sanitized_params['alfa_period'] = \
            int(params['alfa_period'])
    else:
        warnings.warn(
            "No setting for alfa period detected. Using default: 100.",
            RuntimeWarning
        )
        sanitized_params['alfa_period'] = 100

    # Cutoff
    if 'cutoff' in params:
        sanitized_params['cutoff'] = \
            int(params['cutoff'])
    else:
        warnings.warn(
            "No setting for alfa period detected. Using default:"
            f" {int(sanitized_params['alfa_period'] * 50)}.",
            RuntimeWarning
        )
        sanitized_params['cutoff'] = int(sanitized_params['alfa_period'] * 50)

    return sanitized_params


def sanitize_search_space_params(params):
    """Check parameter dictionary for missing values and substitute
    defaults where appropriate.

    Parameters
    ----------
    params : Dict
        Dictionary containing parameters defining the search space.

    Returns
    -------
    Dict
        Sanitized dictionary, with the following keys:
            - `infer_precedence` (bool): Flag indicating whether to infer
              and continuously check (implicit) precedence relations.
            - `fracs` (dict of float): Dictionary indicating the
              probability of selecting each neighborhood operator.
            - `start_solution` (str): String indicating the method of
              generating a starting solution. Either "random" or
              "greedy".
    """
    if params is None:
        params = {}
    sanitized_params = {}

    # Infer precedence
    if 'infer_precedence' in params:
        sanitized_params['infer_precedence'] = \
            params['infer_precedence']
    else:
        warnings.warn(
            "No setting for infering precedences detected. Using default:"
            " False",
            RuntimeWarning
        )
        sanitized_params['infer_precedence'] = False

    # Distribution of neighborhoodoperators
    if 'fracs' in params and \
       ('swap' and 'move' and 'movepair' in params['fracs']) and \
       params['fracs']['swap'] + params['fracs']['move'] + \
       params['fracs']['movepair'] == 1:
        sanitized_params['fracs'] = \
            params['fracs']
    else:
        warnings.warn(
            "No valid setting for neighborhood operator distribution was"
            " detected. Using default: [0.95, 0.025, 0.025].",
            RuntimeWarning
        )
        sanitized_params['fracs'] = {
            "swap": 0.95,
            "move": 0.025,
            "movepair": 0.025
        }

    # Starting solution
    if 'start_solution' in params and \
       params['start_solution'] in ['greedy', 'random']:
        sanitized_params['start_solution'] = \
            params['start_solution']
    else:
        warnings.warn(
            "No valid setting for starting solution detected. Using"
            " default: greedy",
            RuntimeWarning
        )
        sanitized_params['start_solution'] = "greedy"

    return sanitized_params
