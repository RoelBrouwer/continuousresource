"""Utility functions that *may* be reused across simulated annealing
implementations."""


import warnings


def sanitize_params(params):
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
            - `infer_precedence` (bool): Flag indicating whether to infer
              and continuously check (implicit) precedence relations.
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

    return sanitized_params
